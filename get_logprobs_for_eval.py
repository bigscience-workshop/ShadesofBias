import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import datasets
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from config import Config
from huggingface_api import HFEndpointAPI
from metrics.metrics import perplexity
from utils import load_endpoint_url

load_dotenv()
config = Config()

PROMPTS_DATASET = config.prompts_dataset_formatted
ALL_LANGUAGES = config.languages


def _get_complex_key(column_type, model_name):
    """Constructs a string based on the model name and column type.
    This is then used as a dictionary key."""
    model_name_regexed = re.sub("/", "_", model_name)
    key = "_".join([column_type, model_name_regexed])
    return key


def _try_strip_all_strings(x):
    try:
        if isinstance(x, str):
            return x.strip()
    except Exception as e:
        print(f"Error processing: {x}, Error: {e}")
    return x


def compute_model(model_name, data, config, max_new_tokens=1,
                  repetition_penalty=1.0, api_url=None, languages=None,
                  warmup_endpoint=True):
    """Loads the endpoint for the model, prompts it,
    and extracts the logprob for the given prompt."""

    languages = languages if languages is not None else ALL_LANGUAGES

    logger.info(f"Inference API for Model {model_name}")
    # Models may be initialized at:
    # https://ui.endpoints.huggingface.co/LanguageShades/endpoints .
    # The endpoint url then must be put in the .env file
    if api_url is None:
        api_url = load_endpoint_url(model_name)
    # Load the model endpoint
    model_api = HFEndpointAPI(model_name=model_name, config=config,
        answer_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
        api_url=api_url, hf_token=os.environ.get("HF_TOKEN", None), )

    logger.info("Querying model")
    # Wake up the model.
    logprobs, logprobs_answer, success = model_api.query_model("WAKE UP! ",
        pred_method="logprob")
    # Continually ping the model until it responds.
    _loop_model_ping(model_api, success, warmup_endpoint)

    logger.info("Starting inference")

    all_rows = []
    for _, stereotype_dct in data.iterrows():
        stereotype_dct = stereotype_dct.to_dict()
        logger.info(stereotype_dct)

        for language in languages:
            language_code = config.language_codes[language]
            try:
                biased_sentence = stereotype_dct[
                    language_code + "_biased_sentences"]
                biased_template = stereotype_dct[language_code + "_templates"]
                if biased_sentence:
                    logprobs, logprobs_answer, success = model_api.query_model(
                        biased_sentence, pred_method="logprob")
                    logger.debug(logprobs)
                else:
                    continue
            except Exception as e:
                print(e)
                sys.stderr.write("Fix %s\n" % language)
                continue
            # Temporary filter to address None values
            logprob = [x["logprob"] for x in logprobs if
                       x["logprob"] is not None]
            logger.debug(logprob)
            total_logprob = sum(logprob)
            ppl = perplexity(logprob)
            logger.info("Summed logprob %.2f" % total_logprob)

            tokens = [x["text"] for x in logprobs]
            tokens_key = _get_complex_key("tokens", model_name)
            logprob_key = _get_complex_key("logprob", model_name)
            ppl_key = _get_complex_key("ppl", model_name)
            stereotype_dct.update({language_code + "_" + logprob_key: logprob,
                language_code + "_" + tokens_key: tokens,
                language_code + "_" + ppl_key: ppl})
        all_rows.append(stereotype_dct)
    df = pd.DataFrame(all_rows)
    _save_to_json(df, model_name)
    return df


def _save_to_json(df, model_name):
    output_path = Path(f"results_{str(datetime.now().date())}")
    output_path.mkdir(exist_ok=True)
    tmp_file_path = output_path / f"{model_name.replace('/', '_')}.json"
    df.to_json(tmp_file_path)


def _read_from_json(model_name, date_time):
    output_path = Path(f"results_{date_time}")
    tmp_file_path = output_path / f"{model_name.replace('/', '_')}.json"
    if tmp_file_path.exists():
        df = pd.read_json(tmp_file_path)
        return df
    return False


def _loop_model_ping(model_api, success, warmup_endpoint):
    while not success and warmup_endpoint:
        logger.info("Warming up endpoint")
        logprobs, logprobs_answer, success = model_api.query_model("WAKE UP! ",
            pred_method="logprob")
        time.sleep(300)


def run_all_models(config, languages, model_list=None):
    if model_list is None:
        model_list = config.base_model_list
    dataset_name = PROMPTS_DATASET
    data = load_dataset(dataset_name)["test"].to_pandas()
    # Strip whitespace, just in case.
    # (Whitespace will affect tokenization, so best to strip before API calls.)
    data = data.applymap(_try_strip_all_strings)
    # For each model
    for model in model_list:
        try:
            # For when it's the 'list' is actually a dict
            cached_data = model_list[model]
        except:
            cached_data = None
        if cached_data is not None:
            data = _read_from_json(model, cached_data)
        else:
            data = compute_model(model_name=model, data=data, config=config,
                languages=languages)
    return data


if __name__ == "__main__":
    # What languages are we evaluating?
    # languages = ["English", "French"]
    languages = config.languages
    # What models are we evaluating?
    model_dict = {"Qwen/Qwen2-7B": '2024-10-09',
                  "meta-llama/Meta-Llama-3-8B": '2024-10-10',
                  "bigscience/bloom-7b1": None}
    # Where should we write the results to?
    output_hub_dataset_name = "LanguageShades/FormattedBiasShadesWithLogprobs"
    # Call the model endpoints with the prompts in the given languages;
    # sum the logprob and return a new dataset with this information.
    df = run_all_models(config=config, languages=languages,
        model_list=model_dict, )
    hub_dataset = datasets.Dataset.from_pandas(df)
    hub_dataset_dict = datasets.DatasetDict({"test": hub_dataset})
    hub_dataset_dict.push_to_hub(output_hub_dataset_name)
