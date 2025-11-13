import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# def custom_transform(example):
#     ################################
#     ##### YOUR CODE BEGINGS HERE ###

#     # Design and implement the transformation as mentioned in pdf
#     # You are free to implement any transformation but the comments at the top roughly describe
#     # how you could implement two of them --- synonym replacement and typos.

#     # You should update example["text"] using your transformation

#     raise NotImplementedError

#     ##### YOUR CODE ENDS HERE ######

#     return example

def custom_transform(example):
   
    text = example["text"]


    tokens = word_tokenize(text)

    max_replacements = 2
    replacements_done = 0

    idxs = list(range(len(tokens)))
    random.shuffle(idxs)

    for idx in idxs:
        if replacements_done >= max_replacements:
            break

        word = tokens[idx]

        if not word.isalpha():   
            continue
        if len(word) <= 3:
            continue

        synsets = wordnet.synsets(word)
        if not synsets:
            continue

        lemmas = set()
        for syn in synsets:
            for l in syn.lemmas():
                candidate = l.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    lemmas.add(candidate)

        if not lemmas:
            continue

        new_word = random.choice(list(lemmas))

        tokens[idx] = new_word
        replacements_done += 1

    detok = TreebankWordDetokenizer()
    new_text = detok.detokenize(tokens)

    example["text"] = new_text

    return example
