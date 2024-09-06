# -*- coding: utf-8 -*-

import re
from collections import defaultdict

import datasets
from dotenv import load_dotenv

import config

load_dotenv()
from tqdm import tqdm

from utils import get_set
from collections import Counter

config = config.Config()

# TODO: Fix the fact that LKW-Fahrerin and LKW-Fahrer really oughtn't be
# TODO(cont): a false positive for a template slot.
special_words = ["LKW", "AIDS", "SIDA", "LKW-F", "LKW-Fahrerin", "LKW-Fahrer",
                 "ADHD", "TDAH", "EUA", "ADHS", "RPA", "TDHA"]
expected_types = ["AGE", "AGE_STATE", "BODY_HAIRCOLOR", "BODY_HEIGHT", "BODY_SKIN_COLOR",
                  "BODY_WEIGHT", "BODY_WEIGHT_STATE", "CLOTHING_HEAD",
                  "DISABILITY", "ETHNICITY", "FAMILY_REL", "FIELD", "GENDER",
                  "GENDER+AGE", "GENDER_POSS", "GENDER_RELATIVE_PRONOUN",
                  "LANGUAGE", "MENTAL_ISSUE", "MENTAL_ISSUE_ACTIVITY", "NATION",
                  "NATIONALITY", "NATIONALITY_LOCALE", "REGION_PERSON", "REGION", "OCCUPATION",
                  "OCCUPATION+GENDER", "OCCUPATION_TYPE", "PHYSICAL_APPEARANCE",
                  "POLITICAL_IDEA", "POLITICAL_IDENTITY", "SEXUAL_ORIENTATION",
                  "SEXUAL_ORIENTATION_STATE", "SOCIO_EC", "URBANITY"]
expected_mods = ["ADJ", "PL", "1", "2"]
# I no longer recall why this seemed important to separate/mark differently
# from the other mods.
expected_gender = ["MASC", "FEM", "NEUT"]
slot_descriptions = \
    {"AGE": " - Phrases that refer to people by their age.",
     "AGE_STATE": " - Phrases that refer to age. NOT a reference to a person or group of people.",
 "BODY_HAIRCOLOR": " - Phrases that refer to people by their haircolor.",
 "BODY_HEIGHT": "- Phrases that refer to people by their height.",
 "BODY_SKIN_COLOR": " - Phrases that refer to people by their skin color.",
 "BODY_WEIGHT": " - Phrases that refer to people by their weight",
 "BODY_WEIGHT_STATE": " - Phrases that refer to a particular type of body weight. NOT a reference to a person or group of people.",
"CLOTHING_HEAD" : " - Phrases that refer to a piece of clothing on the head. NOT a reference to a person or group of people.",
     "DISABILITY": " - Phrases that refer to people by their ability status.",
 "ETHNICITY": " - Phrases that refer to people by their ethnicity.",
 "FAMILY_REL": " - Phrases that refer to people as their relationship to someone in a family.",
     "FIELD": " - Phrases that refer to an area of study. NOT a reference to a person or group of people.",
 "GENDER": " - Phrases that refer to people by their gender.",
     "GENDER+AGE": " - Phrases that refer to people by their gender and age.",
     "GENDER_RELATIVE_PRONOUN": " - Relative pronouns conveying the gender of the person.",
     "LANGUAGE": " - Phrases that refer to a language. NOT a reference to a person or group of people.",
 "MENTAL_ISSUE": " - Phrases for mental issues.  NOT a reference to a person or group of people.",
     "MENTAL_ISSUE_ACTIVITY": " - Phrases for activities that have to do with mental issues. NOT a reference to a person or group of people.",
 "MENTAL_ISSUE_STATE": " - Phrases for mental issue states.  NOT a reference to a person or group of people.",
 "OCCUPATION": " - Phrases that refer to people by their occupation.",
     "OCCUPATION+GENDER": " - Phrases that refer to people by their gendered occupation.",
     "OCCUPATION_TYPE": " - Phrases for occupations.  NOT a reference to a person or group of people.",
    "PHYSICAL_APPEARANCE": " - Phrases that refer to people by their physical appearance.",
 "POLITICAL_IDEA": " - Phrases for political ideologies.  NOT a reference to a person or group of people.",
 "POLITICAL_IDENTITY": " - Phrases that refer to people by their political ideology.",
     "REGION": " - Phrases that refer to a region. NOT a reference to a person or group of people.",
     "REGION_PERSON": " - Phrases that refer to people by the region they are from.",
 "SEXUAL_ORIENTATION": " - Phrases that refer to people by their sexual orientation.",
     "SEXUAL_ORIENTATION_STATE": " - Phrases that refer to sexual orientation.",
 "SOCIO_EC": " - Phrases that refer to people by their socioeconomic class.",
 "URBANITY": " - Phrases that refer to people by the urban area they live in."}

# Removed:
#"NATION": " - Phrases for nations.  NOT a reference to a person or group of people.",
#"NATIONALITY": " - Phrases that refer to people by their nationality.",
#"NATIONALITY_LOCALE": " - Phrases that refer to people by a locale within a nation.",
# "GENDER_POSS": " - Possessive pronouns conveying the gender of the possessor.",

mod_descriptions = {
    "-ADJ": " - Adjectives/descriptors. Describe singulars, no gender marked.",
    "-PL": " - Plurals, no gender marked.",
    "-ADJ-PL": " - Adjectives/descriptors. Describe plurals, no gender marked.",
    ":FEM-ADJ-PL": " - Adjectives/descriptors. Describe feminine plurals.",
    ":MASC-ADJ-PL": " - Adjectives/descriptors. Describe masculine plurals.",
    ":MASC-ADJ": " - Adjectives/descriptors. Describe masculine singulars.",
    ":MASC-PL": " - Masculine plurals.",
    ":FEM-ADJ": " - Adjectives/descriptors. Describe feminine singulars.",
    ":FEM-PL": " - Feminine plurals.",
    ":FEM": " - Feminine singulars.",
    ":MASC": " - Masculine singulars.",
    ":NEUT": " - Neuter singulars.",
    "": " - Singulars, no gender marked."
}
dataset = datasets.load_dataset("LanguageShades/BiasShadesRaw")['train']
print("Dataset:")
print(dataset)

slot_re = re.compile('([A-Z]{2,}([:+_-]?[A-Z1-9]?)+)')

cols = dataset.column_names
indices = dataset['Index']
subsets = dataset['Subset']
sentence_ids = [x for x in zip(indices, subsets)]

DEBUG = True
def extract_substring(bias_sentence, template_sentence):
    """
    Extracts the substring in a that is not in b.

    Args:
    a: The first string.
    b: The second string.

    Returns:
    The substring in a that is not in b.
    """
    print("Biased sentence: %s" % bias_sentence)
    print("Template: %s" % template_sentence)
    error = False
    normalizer = {"“": "\"", "”": "\"", "’": "'"}
    bias_sentence = ' '.join(bias_sentence.split())
    for key, value in normalizer.items():
        bias_sentence = re.sub(key, value, bias_sentence)
    template_sentence = ' '.join(template_sentence.split())
    # Handling for very common error.
    if bias_sentence[-1] == "." and template_sentence[-1] != ".":
        template_sentence += "."
    for key, value in normalizer.items():
        template_sentence = re.sub(key, value, template_sentence)
    slot_spans = list(slot_re.finditer(template_sentence))
    slot_vocab = defaultdict(list)
    last_slot_spans = []
    i = 1
    while slot_spans != []:
        i += 1
        #print(slot_spans)
        # Don't get stuck in a loop
        if i > 5: #or (last_slot_spans and last_slot_spans[0] == slot_spans[0]):
            #print('\nCheck:')
            #print(bias_sentence)
            #print(template_sentence)
            error = True
            return slot_vocab, error
        last_slot_spans = slot_spans
        slot_span = slot_spans[0]
        # longest match
        slot_name = slot_span.group()
        # Corner case, when a template slot has been replaced by a
        # special word that looks like a slot.
        if slot_name in special_words:
            slot_spans = slot_spans[1:]
            continue
        if DEBUG:
            print("SLOT: %s" % slot_name)
        # The substrings on either side of the slot. These will match
        # in the biased sentence, god willing.
        temp_substrings = [template_sentence[:slot_span.start()],
                           template_sentence[slot_span.end():]]
        # Make it so that we just focus on one slot at a time:
        temp_prev_substring = temp_substrings[0]
        if DEBUG:
            print("First substring: %s" % temp_prev_substring)
        temp_next_substring = temp_substrings[1]
        if DEBUG:
            print("Second substring: %s" % temp_next_substring)
        # Handling for phonetic marker used with slots in some languages
        if temp_next_substring and temp_next_substring[0] == "*":
            temp_next_substring = re.sub("^\*[A-Za-z]*", "",
                                         temp_next_substring)
        # If there are more slots in this sentence, get them out
        # of the alignment; cut the next substring we consider up until
        # that next slot.
        if slot_re.findall(temp_prev_substring):
            temp_prev_substring = slot_re.sub('POOP', temp_prev_substring)
            temp_prev_substring_split = temp_prev_substring.split('POOP')
            temp_prev_substring = temp_prev_substring_split[-1]
        if slot_re.findall(temp_next_substring):
            temp_next_substring = slot_re.sub('POOP', temp_next_substring)
            temp_next_substring_split = temp_next_substring.split('POOP')
            temp_next_substring = temp_next_substring_split[0]
        # Position of the words in the bias sentence *before* the desired vocabulary
        if temp_prev_substring == "":
            # Start of sentence.
            bias_prev_substring_idx = 0
        else:
            bias_prev_substring_idx = bias_sentence.rfind(
                temp_prev_substring) + len(temp_prev_substring)
        # Position of the words in the bias sentence *after* the desired vocabulary
        if temp_next_substring == "":
            # End of sentence.
            bias_next_substring_idx = len(bias_sentence)
        else:
            bias_next_substring_idx = bias_sentence[
                                      bias_prev_substring_idx:].find(
                temp_next_substring) + bias_prev_substring_idx
        # It was not found; error aligning the two statements.
        if bias_next_substring_idx == -1:
            vocab = ""
        else:
            # Vocabulary item from the biased sentence
            vocab = bias_sentence[
                    bias_prev_substring_idx:bias_next_substring_idx]
        if DEBUG:
            print("VOCAB: %s" % vocab)
        if vocab == "":
            print('\nCheck:')
            print(bias_sentence)
            print(template_sentence)
            error = True
        else:
            # Remove the identifiers that distinguish multiple entities
            # in one sentence -- this isn't relevant for the vocabulary.
            slot_name = re.sub('-[1-9]', '', slot_name)
            slot_vocab[slot_name] += [vocab]
        # Fill in this part of the template with the right word,
        # so we don't have to keep handling an already-solved template slot
        # when we move on to the next slot.
        template_sentence = "".join(
            [temp_substrings[0], vocab, temp_substrings[1]])
        if DEBUG:
            print("template sentence is now %s" % template_sentence)
        slot_spans = list(slot_re.finditer(template_sentence))
    return slot_vocab, error

def dissect_slot(slot):
    split_slot = slot.split("-")
    slot_head_tmp = split_slot[0].strip()
    tags = []
    for n in split_slot[1:]:
        tags += n.split(":")
    split_slot_head = slot_head_tmp.split(":")
    slot_head = split_slot_head[0].strip()
    tags += split_slot_head[1:]
    return slot_head, tags

def norm_slot(biased_template, tags):
    error = False
    slot_list = []
    for g in expected_gender:
        if g in tags:
            slot_list += [":" + g]
            tags.remove(g)
    for mod in expected_mods:
        if mod in tags:
            slot_list += ["-" + mod]
            tags.remove(mod)
    if tags != []:
        print("ISSUE WITH TAGS:")
        print(tags)
        print(biased_template)
        error = True
    return "".join(slot_list), error

all_slots = defaultdict(lambda: Counter())
slot_counter = Counter()
for language in config.languages:
    full_slot_vocab = defaultdict(lambda: defaultdict(Counter))
    errors = []
    print("====== LANGUAGE: %s" % language)
    for i, stereotype_dct in enumerate(tqdm(dataset)):
        if language + ": Templates" not in stereotype_dct:
            #print("No template; continuing")
            continue
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
            slot_vocab, error = extract_substring(biased_sentence, biased_template)
            for slot in slot_vocab:
                if slot in special_words:
                    continue
                bare_slot, tags = dissect_slot(slot)
                slot_counter[bare_slot] += 1
                mods, error = norm_slot(biased_template, tags)
                all_slots[bare_slot + mods][language] += 1
                for term in slot_vocab[slot]:
                    full_slot_vocab[bare_slot][mods][term] += 1
            if error:
                errors += [(biased_sentence, biased_template)]

    with open("lexica/" + language + ".txt", "w+") as f:
        f.write("Please check that the following lexicon is correct. If it is not, please update the corresponding template with the correct slot category. For reference, all slot types are listed at the bottom.\n\n")
        for bare_slot, mods in sorted(full_slot_vocab.items()):
            try:
                slot_description = slot_descriptions[bare_slot]
            except KeyError:
                # Fix standardization issue
                if bare_slot == "GENDER+OCCUPATION":
                    bare_slot = "OCCUPATION+GENDER"
                    slot_description = slot_descriptions[bare_slot]
                else:
                    slot_description = " - ERROR. Not a known slot. Typo?\n"
            f.write("\n" + bare_slot + slot_description + "\n")
            for mod in mods:
                f.write("  " + mod + mod_descriptions[mod] + "\n")
                values = full_slot_vocab[bare_slot][mod]
                for value in values:
                    if value == '':
                        continue
                    f.write('\t   ' + value + "\n")
        f.write("\n=== LEXICON ISSUES ===\nThe following stereotype/template pairs couldn't be aligned; is there an error?\n")
        for bias_tuple in errors:
            biased_sentence = bias_tuple[0]
            biased_template = bias_tuple[1]
            f.write(biased_sentence + "\n")
            f.write(biased_template + "\n")
            f.write("\n")
        f.write("\n\n")
        f.write("=== FOR REFERENCE: TEMPLATE SLOT CATEGORIES ===\n")
        f.write("Slot categories are made by concatenating a BASIC SLOT TYPE and MODIFIERS.\n")
        f.write("== BASIC SLOT TYPES: ==\n")
        for slot_type, slot_description in slot_descriptions.items():
            f.write(slot_type + slot_description + "\n")
        f.write("\n== MODIFIERS: ==\n")
        for mod, mod_description in mod_descriptions.items():
            if mod == "":
                f.write(mod + "   (no additional marking; assumed as default)" + mod_description + "\n")
            else:
                f.write(mod + mod_description + "\n")
        f.write("Finally, if there are multiple slots of the same type referring to different people, please add an additional -1, -2 etc., as an additional modifier tag on the template slot category, to disinguish them.")


slots_counts = sorted(all_slots.items())
for slot, languages in slots_counts:
    print(slot)
    print(languages)
