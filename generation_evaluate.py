import os
import re
import sys
import pdb

from datasets import load_dataset
from dotenv import load_dotenv

from metrics.metrics import perplexity

load_dotenv()
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import Config
from huggingface_api import HFEndpointAPI
from utils import get_set, load_endpoint_url, format_single_prompt, helper_parse_for_labels


def main(
    model_name,
    max_new_tokens=1,
    repetition_penalty=1.0,
    dataset_revision=None,
    api_url=None,
    promptparams={
        'prompt_lang': 'en',
        'prompt_name': 'prompt2.4'
    }
):
    config = Config()
    logger.info("Loading dataset")
    data = load_dataset(config.prompts_dataset, revision=dataset_revision)["train"]
    logger.info(f"Inference API for Model {model_name}")
    if api_url is None:
        api_url = load_endpoint_url(model_name)
    model_api = HFEndpointAPI(
        model_name=model_name,
        config=config,
        answer_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        api_url=api_url,
        hf_token=os.environ.get("HF_TOKEN", None),
    )
    logger.info("Starting inference")
    for i, stereotype_dct in enumerate(tqdm(data)):
        logger.info(stereotype_dct)
        id = stereotype_dct["Index"]
        bias_type = stereotype_dct["Bias Type"]
        orig_languages = get_set(stereotype_dct["Original Language of the Stereotype"])
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
        for language in config.languages:
            try:
                biased_sentence = stereotype_dct[language + ": Biased Sentences"]
                if biased_sentence:
                    # Prompt formatting
                    prompt = format_single_prompt(biased_sentence, promptparams)
                    generated_text, success = model_api.query_model(
                        prompt, pred_method='rawgen'
                    )
                else:
                    continue
            except KeyError:
                sys.stderr.write("Fix %s\n" % language)
                continue
            # Parse for lables in generated text
            pred_label = helper_parse_for_labels(generated_text, labels=['yes', 'no'])
            logger.info(f"Predicted Label: {pred_label}")




if __name__ == "__main__":
    main(model_name="bigscience/bloom-7b1",            
         dataset_revision="48897fd", 
         promptparams={
        'prompt_lang': 'en',
        'prompt_name': 'prompt2.4'
    })
