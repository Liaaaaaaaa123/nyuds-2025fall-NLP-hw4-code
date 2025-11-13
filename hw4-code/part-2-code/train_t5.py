import os
import argparse
import torch

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_LEN = 256
# MAX_LEN = 384


def get_args():
    p = argparse.ArgumentParser()

    # model/training
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--from_scratch", action="store_true")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--optimizer_type", type=str, default="AdamW")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
    )
    p.add_argument("--num_warmup_epochs", type=int, default=1)
    p.add_argument("--max_n_epochs", type=int, default=40)
    p.add_argument("--patience_epochs", type=int, default=6)

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--experiment_name", type=str, default="t5_scratch_sqltok_len256")

    # data
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)

    args = p.parse_args()

    # runtime paths
    args.device = DEVICE
    args.results_dir = "results"
    args.records_dir = "records"
    args.experiment_dir = os.path.join("checkpoints", args.experiment_name)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.records_dir, exist_ok=True)
    os.makedirs(args.experiment_dir, exist_ok=True)

    return args


def unpack_batch(batch, device):
    # train/dev: (enc, mask, dec_in, dec_tgt, _)
    if isinstance(batch, (list, tuple)) and len(batch) == 5:
        enc, mask, dec_in, dec_tgt, _ = batch
    else:
        raise ValueError(f"Unexpected batch structure: {type(batch)} / len={len(batch)}")

    enc = enc.to(device)
    mask = mask.to(device)
    dec_in = dec_in.to(device)
    dec_tgt = dec_tgt.to(device)
    return enc, mask, dec_in, dec_tgt


def eval_epoch(
    args,
    model,
    dev_loader,
    tokenizer,
    gt_sql_pth="data/dev.sql",
    gt_record_pkl="records/ground_truth_dev.pkl",
):
    model.eval()
    all_sql = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dev_loader:
            enc, mask, dec_in, dec_tgt = unpack_batch(batch, args.device)

            # #
            # labels = dec_tgt.clone()
            # pad_id = tokenizer.pad_token_id
            # labels[labels == pad_id] = -100 
            # #

            out = model(
                input_ids=enc,
                attention_mask=mask,
                # labels=labels
                labels=dec_tgt,
            )
            total_loss += out.loss.item()

            gen_out = model.generate(
                input_ids=enc,
                attention_mask=mask,
                max_length=MAX_LEN,
                num_beams=5,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,
                # input_ids=enc,
                # attention_mask=mask,
                # max_length=DEC_MAX_LEN,
                # num_beams=DEC_NUM_BEAMS,
                # no_repeat_ngram_size=DEC_NGRAM,
                # length_penalty=DEC_LEN_PENALTY,
                # early_stopping=DEC_EARLY_STOP,
                # decoder_start_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            all_sql.extend(decoded)

    avg_loss = total_loss / len(dev_loader)

    # save dev predictions
    model_sql_path = os.path.join(args.results_dir, f"{args.experiment_name}_dev.sql")
    model_record_path = os.path.join(args.records_dir, f"{args.experiment_name}_dev.pkl")
    save_queries_and_records(all_sql, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_errs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_pkl, model_record_path
    )
    error_rate = len(model_errs) / len(all_sql) if all_sql else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path):
    model.eval()
    all_sql = []
    with torch.no_grad():
        for batch in test_loader:
            # test loader returns: (enc, mask, None, None, None)
            enc = batch[0].to(args.device)
            mask = batch[1].to(args.device)

            gen = model.generate(
                input_ids=enc,
                attention_mask=mask,
                max_length=MAX_LEN,
                num_beams=5,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,
                # input_ids=enc,
                # attention_mask=mask,
                # max_length=DEC_MAX_LEN,
                # num_beams=DEC_NUM_BEAMS,
                # no_repeat_ngram_size=DEC_NGRAM,
                # length_penalty=DEC_LEN_PENALTY,
                # early_stopping=DEC_EARLY_STOP,
                # decoder_start_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            all_sql.extend(decoded)

    save_queries_and_records(all_sql, model_sql_path, model_record_path)


def train(args, model, train_loader, dev_loader, test_loader, tokenizer, optimizer, scheduler):
    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(args.max_n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            enc, mask, dec_in, dec_tgt = unpack_batch(batch, args.device)
            
            # #
            # labels = dec_tgt.clone()
            # pad_id = tokenizer.pad_token_id
            # labels[labels == pad_id] = -100 
            # #

            out = model(
                input_ids=enc,
                attention_mask=mask,
                decoder_input_ids=dec_in,
                # labels=labels
                labels=dec_tgt,
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: train loss = {avg_train_loss:.6f}")

        dev_loss, rec_f1, rec_em, sql_em, err_rate = eval_epoch(
            args, model, dev_loader, tokenizer
        )
        print(
            f"Epoch {epoch}: Dev loss {dev_loss:.6f}, F1 {rec_f1:.6f}, EM {rec_em:.6f}, SQL EM {sql_em:.6f}, SQL err {err_rate*100:.2f}%"
        )

        if rec_f1 > best_f1:
            best_f1 = rec_f1
            bad_epochs = 0
            print(f"New best F1={best_f1:.4f} at epoch {epoch}, saving...")

            save_dir = os.path.join(args.experiment_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)

            best_sql_path = os.path.join(args.results_dir, f"{args.experiment_name}_best_test.sql")
            best_record_path = os.path.join(args.records_dir, f"{args.experiment_name}_best_test.pkl")
            test_inference(args, model, test_loader, tokenizer, best_sql_path, best_record_path)
            print("Saved best test outputs.")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience_epochs:
                print("Early stopping.")
                break

    print(f"Training done. Best F1={best_f1:.4f}")


def main():
    args = get_args()

    if args.use_wandb:
        setup_wandb(args)

    # load data + tokenizer (custom if provided)
    train_loader, dev_loader, test_loader, tok = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        MAX_LEN,
        tokenizer_path=args.tokenizer_path,
    )

    # model
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )

    train(
        args,
        model,
        train_loader,
        dev_loader,
        test_loader,
        tok,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()