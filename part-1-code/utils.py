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


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    text = example["text"]
    words = word_tokenize(text)
    new_words = []

    replace_prob = 0.35

    for word in words:
        if random.random() < replace_prob and word.isalpha():
            synsets = wordnet.synsets(word)
            if synsets:
                lemmas = synsets[0].lemmas()
                if len(lemmas) > 1:
                    synonyms = [lemma.name() for lemma in lemmas if lemma.name().lower() != word.lower()]
                    if synonyms:
                        new_word = random.choice(synonyms).replace('_', ' ')
                        new_words.append(new_word)
                    else:
                        new_words.append(word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
