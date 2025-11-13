import os
from transformers import T5TokenizerFast

DATA_DIR = "data"
MODEL_NAME = "google-t5/t5-small"
MAX_LEN = 256  # same as you used in load_data.py

tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

def read_lines(path):
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines()]

# load data
train_nl = read_lines(os.path.join(DATA_DIR, "train.nl"))
train_sql = read_lines(os.path.join(DATA_DIR, "train.sql"))
dev_nl = read_lines(os.path.join(DATA_DIR, "dev.nl"))
dev_sql = read_lines(os.path.join(DATA_DIR, "dev.sql"))

def get_raw_stats(nl_lines, sql_lines):
    nl_tok_lens = []
    sql_tok_lens = []
    nl_vocab = set()
    sql_vocab = set()

    for ln in nl_lines:
        toks = tokenizer(ln, add_special_tokens=True).input_ids
        nl_tok_lens.append(len(toks))
        for t in toks:
            nl_vocab.add(t)

    for qs in sql_lines:
        toks = tokenizer(qs, add_special_tokens=True).input_ids
        sql_tok_lens.append(len(toks))
        for t in toks:
            sql_vocab.add(t)

    mean_nl = sum(nl_tok_lens) / len(nl_tok_lens)
    mean_sql = sum(sql_tok_lens) / len(sql_tok_lens)

    return {
        "num_examples": len(nl_lines),
        "mean_nl_len": mean_nl,
        "mean_sql_len": mean_sql,
        "nl_vocab_size": len(nl_vocab),
        "sql_vocab_size": len(sql_vocab),
    }

def get_processed_stats(nl_lines, sql_lines, max_len):
    nl_tok_lens = []
    sql_tok_lens = []
    nl_vocab = set()
    sql_vocab = set()

    for ln in nl_lines:
        toks = tokenizer(
            ln,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len
        ).input_ids
        nl_tok_lens.append(len(toks))
        for t in toks:
            nl_vocab.add(t)

    for qs in sql_lines:
        toks = tokenizer(
            qs,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len
        ).input_ids
        sql_tok_lens.append(len(toks))
        for t in toks:
            sql_vocab.add(t)

    mean_nl = sum(nl_tok_lens) / len(nl_tok_lens)
    mean_sql = sum(sql_tok_lens) / len(sql_tok_lens)

    return {
        "mean_nl_len": mean_nl,
        "mean_sql_len": mean_sql,
        "nl_vocab_size": len(nl_vocab),
        "sql_vocab_size": len(sql_vocab),
    }

# before preprocessing
train_raw = get_raw_stats(train_nl, train_sql)
dev_raw = get_raw_stats(dev_nl, dev_sql)

# after preprocessing (truncation etc.)
train_proc = get_processed_stats(train_nl, train_sql, MAX_LEN)
dev_proc = get_processed_stats(dev_nl, dev_sql, MAX_LEN)

print("=== BEFORE PRE-PROCESSING (Table 1) ===")
print("Train:", train_raw)
print("Dev:  ", dev_raw)
print()
print("=== AFTER PRE-PROCESSING (Table 2) ===")
print(f"Model name: {MODEL_NAME}, max_len={MAX_LEN}")
print("Train:", train_proc)
print("Dev:  ", dev_proc)