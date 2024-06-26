import os
import re
import sys
import time
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from map_dataset import convert_dataset
from metrics.metrics import perplexity

load_dotenv()
from datetime import datetime

import numpy as np
import pandas as pd
from config import Config
from huggingface_api import HFEndpointAPI
from loguru import logger
from tqdm import tqdm
from utils import get_set, load_endpoint_url


def get_complex_key(column_type, model_name):
    model_name_regexed = re.sub("/", "_", model_name)
    key = "_".join([column_type, model_name_regexed])
    return key


def compute_model(
    model_name,
    data,
    config,
    max_new_tokens=1,
    repetition_penalty=1.0,
    api_url=None,
    languages=None,
    warmup_endpoint=True,
):
    logger.info("Loading input dataset")

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
    languages = languages if languages is not None else config.languages
    logger.info("Starting inference")
    all_rows = []
    logprobs, logprobs_answer, success = model_api.query_model(
        "test ", pred_method="logprob"
    )

    output_path = Path(f"results_{str(datetime.now().date())}")
    output_path.mkdir(exist_ok=True)
    tmp_file_path = output_path / f"{model_name.replace('/','_')}.json"
    if tmp_file_path.exists():
        df = pd.read_json(tmp_file_path)
        return df

    while not success and warmup_endpoint:
        logger.info("Warming up endpoint")
        logprobs, logprobs_answer, success = model_api.query_model(
            "test ", pred_method="logprob"
        )
        time.sleep(300)

    for _, stereotype_dct in data.iterrows():
        stereotype_dct = stereotype_dct.to_dict()
        logger.info(stereotype_dct)
        # if _ > 5:
        #     continue

        for language in languages:
            try:
                biased_sentence = stereotype_dct[language + ": Biased Sentences"]
                biased_template = stereotype_dct[language + ": Templates"]
                if biased_sentence:
                    logprobs, logprobs_answer, success = model_api.query_model(
                        biased_sentence, pred_method="logprob"
                    )
                    logger.debug(logprobs)
                else:
                    continue
            except Exception as e:
                print(e)
                sys.stderr.write("Fix %s\n" % language)
                continue
            # Temporary filter to address None values
            logprob = [x["logprob"] for x in logprobs if x["logprob"] is not None]
            logger.debug(logprob)
            total_logprob = sum(logprob)
            mean_logprob = np.mean(logprob)
            n_tokens = len(logprob)
            ppl = perplexity(logprob)
            logger.info("Summed logprob %.2f" % total_logprob)

            tokens = [x["text"] for x in logprobs]
            tokens_key = get_complex_key("tokens", model_name)
            logprob_key = get_complex_key("logprob", model_name)
            stereotype_dct.update(
                {
                    language + "_" + logprob_key: logprob,
                    language + "_" + tokens_key: tokens,
                }
            )
        all_rows.append(stereotype_dct)
    df = pd.DataFrame(all_rows)
    df.to_json(tmp_file_path)
    return df


def try_strip_all_strings(x):
    try:
        if isinstance(x, str):
            return x.strip()
    except Exception as e:
        print(f"Error processing: {x}, Error: {e}")
    return x


def run_all_models(config, languages, output_hub_dataset_name, model_list=None):
    if model_list is None:
        model_list = config.base_model_list
    dataset_name = config.prompts_dataset
    data = load_dataset(dataset_name)["train"].to_pandas()
    # Will affect tokenization, so best to strip before API calls
    data = data.applymap(try_strip_all_strings)
    for models in model_list:
        data = compute_model(
            model_name=models, data=data, config=config, languages=languages
        )
    df = convert_dataset("BiasShades_fields - columns.csv", df=data)
    df = datasets.Dataset.from_pandas(df)
    df = datasets.DatasetDict({"test": df})
    df.push_to_hub(output_hub_dataset_name)


if __name__ == "__main__":
    config = Config()
    languages = [
        "English",
        "French",
    ]
    model_list = ["Qwen/Qwen2-7B", "meta-llama/Meta-Llama-3-8B", "bigscience/bloom-7b1"]
    output_hub_dataset_name = "LanguageShades/FormattedBiasShades"
    run_all_models(
        config=config,
        languages=languages,
        output_hub_dataset_name=output_hub_dataset_name,
        model_list=model_list,
    )
