import os
import pdb
import re


def load_endpoint_url(model_name):
    env_var_key = "MODEL_" + re.sub(r"\W", "_", model_name).upper()
    endpoint_url = os.getenv(env_var_key)
    if endpoint_url:
        return endpoint_url
    else:
        raise ValueError(
            f"Endpoint URL for {model_name} not found in environment variables"
        )


def get_set(entry):
    if entry:
        return [i.strip() for i in entry.split(",")]


def format_single_prompt(biased_sentence, promptparams, prompt_lang):
    """
    biased_sentence: A string of biased sentence
    promptparams: A dict of parameters for prompt curation
        'prompt_lang' - language code of the corresponding prompt
        'prompt_name' - the name of the prompt, should correspond to a .txt file in prompts/prompts_{prompt_lang}
    returns: A string of formatted prompt
    """
    # Load the corresponding prompt template
    prompt_file = f"prompts/prompts_{prompt_lang}/{promptparams['prompt_name']}.txt"
    curr_prompt_file = open(prompt_file, "r")
    prompt_template = curr_prompt_file.read()
    curr_prompt_file.close()
    # Format the prompt
    return prompt_template.format(input=biased_sentence)


def helper_parse_for_labels(text, prompt_type):
    """
    text: text returned by inference
    labels: A list of available labels for the current format. The order
        has to be correspond to the labels
    Returns: The corresponding label of the generated text. Return None if
        model failed to generate.
    """

    for label in ['نعم', 'হ্যাঁ', 'sim', '是', 'ja', 'yes', 'oui', 'हाँ', 'sì', 'tak', 'da', 'да', 'होय']:
        if label in text.lower():
            if prompt_type == 'final_prompt3':
                return 0
            else:
                return 1
    for label in ["لا", 'না', 'não', '否', 'nee', 'no', 'non', 'nein', 'नहीं', 'nie', 'nu', 'нет', 'नाही']:
        if label in text.lower():
            if prompt_type == 'final_prompt3':
                return 1
            else:
                return 0

    # A failed prediction
    return -1
