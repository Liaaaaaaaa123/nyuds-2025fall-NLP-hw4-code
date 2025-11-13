import os
import json
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config

MODEL_NAME = "google-t5/t5-small"


def initialize_model(args):
    import json, os
    from transformers import T5Config, T5ForConditionalGeneration

    base_name = "google-t5/t5-small"
    if getattr(args, "from_scratch", False):
        config = T5Config.from_pretrained(base_name)

        # adjust vocab
        if args.tokenizer_path:
            tok_json = os.path.join(args.tokenizer_path, "tokenizer.json")
            with open(tok_json, "r") as f:
                tok_data = json.load(f)
            vocab_size = len(tok_data["model"]["vocab"])
            config.vocab_size = vocab_size

        model = T5ForConditionalGeneration(config)

        # OPTIONAL: copy the input embedding from pretrained t5 into ours
        # to give it a better start
        if getattr(args, "init_from_pretrained_embed", False):
            pt = T5ForConditionalGeneration.from_pretrained(base_name)
            with torch.no_grad():
                n = min(pt.shared.weight.size(0), model.shared.weight.size(0))
                model.shared.weight[:n].copy_(pt.shared.weight[:n])
                model.encoder.embed_tokens.weight[:n].copy_(pt.encoder.embed_tokens.weight[:n])
                model.decoder.embed_tokens.weight[:n].copy_(pt.decoder.embed_tokens.weight[:n])

        print("Initialized scratch T5-small (custom vocab, optional pretrained embeds).")
    else:
        model = T5ForConditionalGeneration.from_pretrained(base_name)
        print("Loaded pretrained T5-small for finetuning.")

    model.to(args.device)
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(
        model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not implemented.")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def setup_wandb(args):
    """
    Tiny helper so train_t5.py can call it.
    Will only actually init if user passed --use_wandb.
    """
    try:
        import wandb
    except ImportError:
        print("[t5_utils] wandb not installed; skipping wandb init.")
        return
    wandb.init(project="hw4-t5", name=args.experiment_name)
    wandb.config.update(vars(args))