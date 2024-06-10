import sys
import os
from tqdm import tqdm
import requests
from datasets import load_dataset
from transformers import AutoTokenizer

# When using Spaces, the SHADES_TOKEN will be read from the Settings.
# When running on a local machine, it will be stored in a directory;
# Change this directory as fits your own setup.
TOKEN_STORAGE = "/Users/margaretmitchell/.huggingface/token"
if os.path.exists(TOKEN_STORAGE):
    SHADES_TOKEN = open(TOKEN_STORAGE, "r").readline()
else:
    SHADES_TOKEN = os.environ.get("SHADES_TOKEN")

# API URL for basic I/O is available at https://ui.endpoints.huggingface.co/
ENDPOINT_API = "https://api-inference.huggingface.co/models/"
PROMPTS_DATASET = "LanguageShades/BiasShades"
LANGUAGES = ["Arabic", "Bengali", "Brazilian Portuguese", "Chinese", "Traditional Chinese", "Dutch", "English", "French", "German", "Hindi", "Italian", "Marathi", "Polish", "Romanian", "Russian", "Spanish"]

# Class to manage models deployed as HF endpoints
class HFEndpointAPI:
    def __init__(self, model_name, api_url, hf_token, answer_tokens, repetition_penalty):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # Inference Endpoints for large models
        if model_name == "bigscience/bloom-7b1":
            self.api_url = "https://w3xj0fk1w4zqcb3m.us-east-1.aws.endpoints.huggingface.cloud"
        else:
            self.api_url = api_url + model_name
        self.hf_token = hf_token
        self.answer_tokens = answer_tokens
        self.repetition_penalty = repetition_penalty

    def query(self, payload, api_url):
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        response = requests.post(api_url, headers=headers, json=payload)
        return response.json()

    def endpoint_generate(self, prompt, api_url, n_tokens=1, repetition_penalty=1.0):
        q = self.query(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": n_tokens,
                    "repetition_penalty": repetition_penalty,
                    "decoder_input_details": True,
                },
            },
            api_url,
        )
        if isinstance(q, dict):
            success = False
            return q['error'], success
        else:
            success = True
            return q, success

    def query_model(self, prompt):
        # Remember there will be an error if the model hasn't been pinged in the
        # past hour -- it goes to sleep, then will take a few minutes after you
        # ping it again to wake up.
        try:
            response, success = self.endpoint_generate(prompt, self.api_url)
        except OSError as e:
            response = e
            success = False
        if success:
            logprobs = response[0]['details']["prefill"],
            return logprobs, success
        else:
            print("===== !! Failed to get model response")
            print(response)
            return response, success

def query_model(model_api, prompt):
    print("=== Querying with prompt:")
    print(prompt)
    logprobs, success = model_api.query_model(prompt)
    return logprobs, success

def get_set(entry):
    if entry:
        return [i.strip() for i in entry.split(',')]

def main(model, max_new_tokens=1, repetition_penalty=1.0):
    print("Loading dataset")
    # TODO: Change this in dataset to 'test'
    loaded_dset = load_dataset(PROMPTS_DATASET)['train']
    print("Loading model")
    model_api = HFEndpointAPI(
            model,
            ENDPOINT_API,
            SHADES_TOKEN,
            max_new_tokens,
            repetition_penalty
        )
    print("Starting model queries")
    for i, stereotype_dct in enumerate(tqdm(loaded_dset)):
        print(stereotype_dct)
        id = stereotype_dct["Index"]
        bias_type = stereotype_dct["Bias Type"]
        orig_languages = get_set(stereotype_dct["Original Language of the Stereotype"])
        lang_validity = get_set(stereotype_dct["Language Validity (In which languages is this stereotype valid?)"])
        region_validity = get_set(stereotype_dct["Region Validity (In which regions is this stereotype valid?)"])
        stereotyped_group = stereotype_dct["Stereotyped Group"]
        for language in LANGUAGES:
            try:
                biased_sentence = stereotype_dct[language + ": Biased Sentences"]
                if biased_sentence:
                    logprobs, success = query_model(model_api, biased_sentence)
                    print(logprobs)
                else:
                    continue
            except KeyError:
                sys.stderr.write("Fix %s\n" % language)
                continue
            prompt_logprob = sum(x['logprob'] for x in logprobs[0][1:])
            print("Summed logprob %.2f" % prompt_logprob)

if __name__ == "__main__":
    main("bigscience/bloom-7b1", max_new_tokens=1, repetition_penalty=1)