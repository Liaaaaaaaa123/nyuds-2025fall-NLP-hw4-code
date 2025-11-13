import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

TASK_PREFIX = "translate English to SQL: "
PAD_ID = 0   # we'll set from tokenizer later
MAX_LEN_DEFAULT = 256

# # # Q8
# # import os, re

# TASK_PREFIX = "Translate English question to SQLite SQL. "

# SCHEMA_HINT = (
#     "Tables: flight, airport, city, days. "
# )

# JOIN_HINT = (
#     "Hint: flight.from_airport -> airport.airport_code. "
# )

# PAD_ID = 0   # we'll set from tokenizer later
# MAX_LEN_DEFAULT = 256


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class Text2SQLDataset(Dataset):
    def __init__(self, nl_path: str, sql_path: str = None):
        self.nl = _read_lines(nl_path)
        self.sql = _read_lines(sql_path) if sql_path is not None else None
        if self.sql is not None:
            assert len(self.nl) == len(self.sql)

    def __len__(self):
        return len(self.nl)

    def __getitem__(self, idx):
        item = {"nl": self.nl[idx]}
        if self.sql is not None:
            item["sql"] = self.sql[idx]
        return item


def make_tokenizer(tokenizer_path: str = None):
    if tokenizer_path is None:
        tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        return tok

    tok_json = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(tok_json):
        tok = T5TokenizerFast(tokenizer_file=tok_json)
        # set specials — must match what we used in build_tokenizer.py
        tok.pad_token = "<pad>"
        tok.eos_token = "</s>"
        tok.unk_token = "<unk>"
        tok.bos_token = "<s>"
        return tok
    else:
        # fallback to HF
        return T5TokenizerFast.from_pretrained("google-t5/t5-small")

# def make_tokenizer(tokenizer_path: str = None):
#     # 如果不给路径，就用原始 t5-small
#     if tokenizer_path is None:
#         return T5TokenizerFast.from_pretrained("google-t5/t5-small")

#     # 否则走 sqltok 目录（你已经有 tokenizer.json 了）
#     tok_json = os.path.join(tokenizer_path, "tokenizer.json")
#     if os.path.exists(tok_json):
#         tok = T5TokenizerFast(tokenizer_file=tok_json)
#         # 和 build_tokenizer.py 里保持一致的 special tokens
#         tok.pad_token = "<pad>"
#         tok.eos_token = "</s>"
#         tok.unk_token = "<unk>"
#         tok.bos_token = "<s>"
#         return tok
#     else:
#         # 兜底：路径不对就退回 t5-small，防止直接崩
#         return T5TokenizerFast.from_pretrained("google-t5/t5-small")


def _train_dev_collate(batch, tokenizer: T5TokenizerFast, max_len: int):
    enc_ids = []
    enc_mask = []
    dec_in_ids = []
    dec_tgt_ids = []

    pad_id = tokenizer.pad_token_id

    for item in batch:
        src_text = TASK_PREFIX + item["nl"]

        # # Q8
        # src_text = TASK_PREFIX + SCHEMA_HINT + JOIN_HINT + item["nl"]


        enc = tokenizer(
            src_text,
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )
        enc_ids.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        enc_mask.append(torch.tensor(enc["attention_mask"], dtype=torch.long))

        tgt_text = item["sql"]
        tgt_ids = tokenizer(
            tgt_text,
            max_length=max_len,
            truncation=True,
        )["input_ids"]

        # decoder_input_ids = pad + tgt
        decoder_input = [pad_id] + tgt_ids
        # labels = tgt + eos
        decoder_target = tgt_ids + [tokenizer.eos_token_id]

        dec_in_ids.append(torch.tensor(decoder_input, dtype=torch.long))
        dec_tgt_ids.append(torch.tensor(decoder_target, dtype=torch.long))

    enc_padded = pad_sequence(enc_ids, batch_first=True, padding_value=pad_id)
    mask_padded = pad_sequence(enc_mask, batch_first=True, padding_value=0)
    dec_in_padded = pad_sequence(dec_in_ids, batch_first=True, padding_value=pad_id)
    dec_tgt_padded = pad_sequence(dec_tgt_ids, batch_first=True, padding_value=pad_id)

    init_dec = torch.full((enc_padded.size(0), 1), pad_id, dtype=torch.long)

    return enc_padded, mask_padded, dec_in_padded, dec_tgt_padded, init_dec


def _test_collate(batch, tokenizer: T5TokenizerFast, max_len: int):
    enc_ids = []
    enc_mask = []
    pad_id = tokenizer.pad_token_id
    for item in batch:
        src_text = TASK_PREFIX + item["nl"]
        # # Q8
        # src_text = TASK_PREFIX + SCHEMA_HINT + JOIN_HINT + item["nl"]
        
        enc = tokenizer(
            src_text,
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )
        enc_ids.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        enc_mask.append(torch.tensor(enc["attention_mask"], dtype=torch.long))

    enc_padded = pad_sequence(enc_ids, batch_first=True, padding_value=pad_id)
    mask_padded = pad_sequence(enc_mask, batch_first=True, padding_value=0)
    return enc_padded, mask_padded, None, None, None


def load_t5_data(
    batch_size: int,
    test_batch_size: int,
    max_len: int = MAX_LEN_DEFAULT,
    tokenizer_path: str = None,
):
    tok = make_tokenizer(tokenizer_path)

    train_ds = Text2SQLDataset("data/train.nl", "data/train.sql")
    dev_ds = Text2SQLDataset("data/dev.nl", "data/dev.sql")
    test_ds = Text2SQLDataset("data/test.nl", None)  # test sql is hidden

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _train_dev_collate(b, tok, max_len),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _train_dev_collate(b, tok, max_len),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=lambda b: _test_collate(b, tok, max_len),
    )

    return train_loader, dev_loader, test_loader, tok




