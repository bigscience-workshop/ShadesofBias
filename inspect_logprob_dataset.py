from datasets import load_dataset


ds = load_dataset('LanguageShades/FormattedBiasShadesWithLogprobs')['test']
# can also call config.language_code_list
LANGUAGES = ["ar", "bn", "pt_br", "zh", "zh_hant", "nl", "en", "fr", "de", "hi", "it", "mr", "pl", "ro", "ru", "es"]

for line in ds:
    # Id for the statement: Both the stereotype and contrasts
    index = line['index']
    # If the value is '_original', then it is the original stereotype.
    # Anything else is a contrast, a non-stereotype.
    subset = line['subset']
    # The type of stereotype it is: age, gender, etc
    bias_type = line['bias_type']
    # The language codes where, in the corresponding regions, the statement might be considered true by some.
    # NOTE: It is useful for some experiments to have statements limited by this, so they are ONLY
    # the ones that are actually observed in the given language/region.
    stereotype_valid_langs = line['stereotype_valid_langs']
    # Which characteristics are being stereotyped in this?
    # NOTE: Multiple characteristics may be listed here.
    # intersections are marked with an intersection symbol, e.g., 'girls' is females ∩ children.
    # multiple entities are separated with 'and'
    stereotyped_entity = line['stereotyped_entity']
    # What kind of stereotype is this?
    type = line['type']
    for lang in LANGUAGES:
        # Is this an expression in the given language region? That will probably affect its likelihood.
        is_expression = line[lang + "_expression"]
        for model in ['meta-llama_Meta-Llama-3-8B', 'bigscience_bloom-7b1', 'Qwen_Qwen2-7B']:
            logprob_column_name = '_'.join([lang, 'logprob', model])
            logprobs = line[logprob_column_name]
            if subset == "_original":
                print("Retrieved the following variables for original statement %s:" % index)
            else:
                print("Retrieved the following variables for contrast to statement %s:" % index)
            print("Model: %s" % model)
            print("Language: %s" % lang)
            print("Logprobs: ", end='')
            print(logprobs)

