import os
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from loguru import logger

from config import Config
from huggingface_api import HFEndpointAPI
from metrics.metrics import perplexity
from utils import load_endpoint_url

load_dotenv()
config = Config()

PROMPTS_DATASET = "LanguageShades/FormattedBiasShadesWithLogprobsUpdatedMeta"#"LanguageShades/FormattedBiasShadesWithLogprobsUpdated"#"LanguageShades/FormattedBiasShadesWithLogprobs"#config.prompts_dataset_formatted
ALL_LANGUAGES = config.languages
DATASET_COLUMNS = config.formatted_dataset_columns


def _get_column_name(model_name, column_type=None):
    """Constructs a string based on the model name and column type.
    This is then used as a dictionary key."""
    flat_model_name = re.sub("/", "_", model_name)
    if column_type:
        key = "_".join([column_type, flat_model_name])
        return key
    return flat_model_name


def _try_strip_all_strings(x):
    try:
        if isinstance(x, str):
            return x.strip()
    except Exception as e:
        print(f"Error processing: {x}, Error: {e}")
    return x


def _save_to_json(df, model_name):
    output_path = Path(f"results_{str(datetime.now().date())}")
    output_path.mkdir(exist_ok=True)
    flat_model_name = _get_column_name(model_name)
    tmp_file_path = output_path / f"{flat_model_name}.json"
    df.to_json(tmp_file_path)


def _read_from_json(model_name, date_time, data_df):
    output_path = Path(f"results_{date_time}")
    flat_model_name = _get_column_name(model_name)
    tmp_file_path = output_path / f"{flat_model_name}.json"
    if tmp_file_path.exists():
        js_dict = json.load(open(tmp_file_path, 'r'))
        print(js_dict['fr_logprob_Qwen_Qwen2-7B'])
        model_cols = [key for key in js_dict if flat_model_name in key]
        print(model_cols)
        new_data_df = pd.DataFrame(js_dict)
        #merged_data_df = pd.merge(data_df, new_data_df[DATASET_COLUMNS + model_cols],
        #                     on=DATASET_COLUMNS, how='left')
        # The json may have the wrong type values for the cells.
        new_data_df.loc[new_data_df['index'] == '237/0', 'index'] = 237.0
        data_df.loc[data_df['index'] == '237/0', 'index'] = 237.0
        #print(new_data_df[new_data_df['subset'] == '237/0', 'subset'])
        # This converts to what's expected from the original dataset.
        #for column in DATASET_COLUMNS:
        #    if data_df[column].dtype != new_data_df[column].dtype:
        #            new_data_df[column] = new_data_df[column].astype(data_df[column].dtype)
        #merged_data_df = pd.merge(data_df, new_data_df[DATASET_COLUMNS + model_cols], on=DATASET_COLUMNS, how='left')#, suffixes=('', '_drop'))
        #merged_data_df = merged_data_df[[col for col in merged_data_df.columns if not col.endswith('_drop')]]
        #print(merged_data_df)
        return new_data_df
        #return merged_data_df
    print("Couldn't find tmp file path. Returning dataset unchanged.")
    return data_df


def _loop_model_ping(model_api, success, warmup_endpoint):
    while not success and warmup_endpoint:
        logger.info("Warming up endpoint")
        logprobs, logprobs_answer, success = model_api.query_model("WAKE UP! ",
            pred_method="logprob")
        time.sleep(240)


def compute_model(model_name, data_df, config, max_new_tokens=1,
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
    for _, stereotype_dct in data_df.iterrows():
        stereotype_dct = stereotype_dct.to_dict()
        try:
            del stereotype_dct['__index_level_0__']
        except KeyError:
            pass
        #logger.info(stereotype_dct)
        for language in languages:
            print(language)
            language_code = config.language_codes[language]
            tokens_key = language_code + "_" +  _get_column_name(model_name, "tokens")
            logprob_key = language_code + "_" + _get_column_name(model_name, "logprob")
            ppl_key = language_code + "_" + _get_column_name(model_name, "ppl")
            if logprob_key not in stereotype_dct or stereotype_dct[logprob_key] is None or not stereotype_dct[logprob_key].any():
                if language_code + "_biased_sentences" in stereotype_dct:
                    biased_sentence = stereotype_dct[language_code + "_biased_sentences"]
                    if biased_sentence:
                        print("Did NOT find results for:")
                        print(
                            stereotype_dct[language_code + "_biased_sentences"])

                        logprobs, logprobs_answer, success = model_api.query_model(
                            biased_sentence, pred_method="logprob")
                        logger.debug(logprobs)
                        # Quick filter to address None values
                        logprob = [x["logprob"] for x in logprobs if x["logprob"] is not None]
                        #logger.debug(logprob)
                        total_logprob = sum(logprob)
                        ppl = perplexity(logprob)
                        #logger.info("Summed logprob %.2f" % total_logprob)
                        tokens = [x["text"] for x in logprobs]
                        stereotype_dct.update({logprob_key: logprob, tokens_key: tokens, ppl_key: ppl})
            else:
                print("Found reults for:")
                print(stereotype_dct[language_code + "_biased_sentences"])
        all_rows.append(stereotype_dct)
    df = pd.DataFrame(all_rows)
    _save_to_json(df, model_name)
    return df

def run_all_models(config, languages, model_list=None):
    if model_list is None:
        model_list = config.base_model_list
    dataset_name = PROMPTS_DATASET
    data_df = load_dataset(dataset_name)["test"].to_pandas()
    # Strip whitespace, just in case.
    # (Whitespace will affect tokenization, so best to strip before API calls.)
    data_df = data_df.applymap(_try_strip_all_strings)
    # For each model
    for model in model_list:
        data_df = compute_model(model_name=model, data_df=data_df,
                                config=config, languages=languages)
    return data_df


if __name__ == "__main__":
    # What languages are we evaluating?
    # languages = ["English", "French"]
    languages = config.languages
    # What models are we evaluating?
    # If you don't have the corresponding cached results files listed below,
    # (2024-10-09 etc.), you'll need to recompute.
    # Just make model_dict a list of models,
    # or else set the values in the dict to None.
    #model_dict = {#"Qwen/Qwen2-1.5B": "2024-10-13",
    #              "Qwen/Qwen2-7B": "2024-10-08",}
    #              #"meta-llama/Meta-Llama-3-8B": "2024-10-13",
    #              #"bigscience/bloom-7b1": "2024-10-13",
    #              #"Qwen/Qwen2-72B": "2024-10-13"}
    model_list = ["bigscience/bloom-7b1","bigscience/bloom-1b7", "mistralai/Mistral-7B-v0.1"]#["meta-llama/Meta-Llama-3-70B", "meta-llama/Meta-Llama-3-8B"] #["Qwen/Qwen2-1.5B","Qwen/Qwen2-7B","Qwen/Qwen2-72B",]#["bigscience/bloom-7b1"] #["meta-llama/Meta-Llama-3-70B"]
    # Where should we write the results to?
    output_hub_dataset_name = "LanguageShades/FormattedBiasShadesWithLogprobsUpdatedBloomMistral"
    # Call the model endpoints with the prompts in the given languages;
    # sum the logprob and return a new dataset with this information.
    df = run_all_models(config=config, languages=languages, model_list=model_list)
    print(df)
    print("Done")
    hub_dataset = Dataset.from_pandas(df)
    hub_dataset_dict = DatasetDict({"test": hub_dataset})
    hub_dataset_dict.push_to_hub(output_hub_dataset_name)
