import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def read_all_texts(data_dir):
    texts = []
    for name in ["train.nl", "train.sql", "dev.nl", "dev.sql", "test.nl"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="local_tokenizer")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    texts = read_all_texts(args.data_dir)
    print(f"Collected {len(texts)} lines for tokenizer training")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[
            "<pad>",
            "</s>",
            "<s>",
            "<unk>",
        ],
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    # save a tiny fake "config" so we know pad/eos later
    with open(os.path.join(args.out_dir, "special_tokens_map.txt"), "w") as f:
        f.write("<pad>\n</s>\n<s>\n<unk>\n")

    print(f"Tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()