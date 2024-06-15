# -*- coding: utf-8 -*-
import pandas as pd
import os
from typing import List
import datasets
import config
import collections

import os
import re
import sys

from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv

from metrics.metrics import perplexity

load_dotenv()
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import Config
from huggingface_api import HFEndpointAPI
from utils import get_set, load_endpoint_url
from collections import Counter

config = config.Config()

dataset = datasets.load_dataset("LanguageShades/BiasShades")['train']
print(dataset)

slot_re = re.compile('([A-Z]{2,}([:+_-]?[A-Z1-9]?)+)')

# Set language to check
lang = "English"
templates_lang_col = f"{lang}: Templates"
biased_sentence_lang_col = f"{lang}: Biased Sentences"

print(dataset['Index'])
cols = dataset.column_names
#print(cols)
#df_target_tab = df_target_tab.rename(columns={c: c.strip() for c in cols})
#index_col = "Index" if "Index" in df_target_tab.columns else df_target_tab.columns[0]
indices = dataset['Index']
subsets = dataset['Subset']
sentence_ids = [x for x in zip(indices, subsets)]
print(sentence_ids)

def extract_substring(bias_sentence, template_sentence):
    """
    Extracts the substring in a that is not in b.

    Args:
    a: The first string.
    b: The second string.

    Returns:
    The substring in a that is not in b.
    """
    #bias_sentence = "Men are from mars, women are from venus."
    #template_sentence = "GENDER-PL are from mars, GENDER-PL are from venus."
    normalizer = {"“":"\"", "”":"\"", "’":"'"}

    #bias_sentence = "You're a man and you don't have a say over your wife?"
    #template_sentence = "You're a GENDER-1 and you don't have a say over your GENDER-2_SPOUSE?"
    bias_sentence = ' '.join(bias_sentence.split())
    for key, value in normalizer.items():
        bias_sentence = re.sub(key, value, bias_sentence)
    template_sentence = ' '.join(template_sentence.split())
    for key, value in normalizer.items():
        template_sentence = re.sub(key, value, template_sentence)
    #print(bias_sentence)
    #print(template_sentence)
    # Currently only handling one match
    slot_names_iter = slot_re.finditer(template_sentence)
    #print([(m.start(0), m.end(0)) for m in slot_names_iter])
    #print(slot_names_iter)
    slot_vocab = {}
    j = -1
    for slot_span in slot_names_iter:
        j += 1
        # longest match
        slot_name = slot_span.group()
        #print("SLOT: %s" % slot_name)
        # The substrings on either side of the slot. These will match
        # in the biased sentence, god willing.
        substrings = template_sentence.split(slot_name)
        # Make it so that we just focus on one slot at a time.
        substrings = [slot_re.sub('POOP', substr) for substr in substrings]
        new_substrings = []
        for substring in substrings:
            split_substring = substring.split('POOP')
            new_substrings += split_substring
        substrings = new_substrings

        prev_sentence = ""
        vocab_list = []
        for i in range(len(substrings)-1):
            first_substring = substrings[i]
            prev_sentence += first_substring
            #print("First substring: %s" % first_substring)
            next_substring = substrings[i+1]
            #print("Second substring: %s" % next_substring)
            next_substring_idx = bias_sentence.find(next_substring)
            vocab = bias_sentence[len(prev_sentence):next_substring_idx]
            prev_sentence += vocab
            #print("VOCAB: %s" % vocab)
            vocab_list += [vocab]
        final_vocab = vocab_list[j]
        if final_vocab == '':
            print('\nCheck:')
            print(bias_sentence)
            print(template_sentence)
        #print("VOCAB: %s" % final_vocab)
        slot_name = re.sub('-[1-9]', '', slot_name)
        if slot_name not in slot_vocab:
            slot_vocab[slot_name] = {final_vocab.lower():1}
        else:
            if final_vocab.lower() not in slot_vocab[slot_name]:
                slot_vocab[slot_name][final_vocab.lower()] = 1
            else:
                slot_vocab[slot_name][final_vocab.lower()] += 1
    return slot_vocab

regional_stereotypes = {}
for language in config.languages:
    full_slot_vocab = {}
    print("====== LANGUAGE: %s" % language)
    for i, stereotype_dct in enumerate(tqdm(dataset)):
        #logger.info(stereotype_dct)
        index = stereotype_dct["Index"]
        subset = stereotype_dct["Subset"]
        bias_type = stereotype_dct["Bias Type"]
        orig_languages = get_set(
            stereotype_dct["Original Language of the Stereotype"])
        lang_validity = get_set(
            stereotype_dct[
                "Language Validity (In which languages is this stereotype valid?)"
            ]
        )
        region_validity = get_set(
            stereotype_dct[
                "Region Validity (In which regions is this stereotype valid?)"
            ]
        )
        stereotyped_group = stereotype_dct["Stereotyped Group"]
        biased_sentence = stereotype_dct[language + ": Biased Sentences"]
        # Templates repeat in numerical order; this fills-in-blanks
        if stereotype_dct[language + ": Templates"] != "":
            biased_template = stereotype_dct[language + ": Templates"]
        is_expression = stereotype_dct[language + ": Is this a saying?"]
        comments = stereotype_dct[language + ": Comments"]
        if biased_template and biased_sentence:
            slot_vocab = extract_substring(biased_sentence, biased_template)
            #print(slot_vocab)
            #print(full_slot_vocab)
            for slot in slot_vocab:
                for word in slot_vocab[slot]:
                    try:
                        full_slot_vocab[slot][word] += 1
                    except KeyError:
                        try:
                            full_slot_vocab[slot][word] = 1
                        except KeyError:
                            full_slot_vocab[slot] = {word: 1}

    for slot, values in sorted(full_slot_vocab.items()):
        print(slot)
        for value in values:
            if value == '':
                continue
            print('\t' + value)
