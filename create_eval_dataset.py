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



def create_dict(index, subset, bias_type, orig_languages, lang_validity, region_validity, stereotyped_group, logprob_dict):
    eval_dict = {
        'index': index,
        'subset': subset,
        'bias_type': bias_type,
        'stereotype_origin_langs': orig_languages,
        'stereotype_valid_langs': lang_validity,
        'stereotype_valid_regions': region_validity,
        'stereotyped_group': stereotyped_group
    }
    for lang_code, keys in logprob_dict.items():
        for key in keys:
            eval_dict[lang_code + "_" + key] = logprob_dict[lang_code][key]
    return eval_dict


def get_complex_key(column_type, model_name):
    model_name_regexed = re.sub('/', '_', model_name)
    key = '_'.join([column_type, model_name_regexed])
    return key

def init_eval_dataset(config, model_name):
    columns = [
        'index',
        'subset',
        'bias_type',
        'stereotype_origin_langs',
        'stereotype_valid_langs',
        'stereotype_valid_regions',
        'stereotyped_group']
    for language in config.languages:
        lang_code = config.language_codes[language]
        columns += [lang_code + '_biased_sentence']
        columns += [lang_code + '_biased_template']
        columns += [lang_code + '_is_expression']
        columns += [lang_code + '_comments']
        columns += [lang_code + '_' + get_complex_key('tokens', model_name)]
        columns += [lang_code + '_' + get_complex_key('logprob', model_name)]
    print("Created:")
    print(columns)
    columns_dict = {key: [] for key in columns}
    eval_dataset = DatasetDict({'test': Dataset.from_dict(columns_dict)})
    return eval_dataset


def main(
    model_name,
    max_new_tokens=1,
    repetition_penalty=1.0,
    dataset_revision=None,
    api_url=None,
):
    config = Config()
    eval_dataset = init_eval_dataset(config, model_name)
    print("Initialized output dataset:")
    print(eval_dataset)
    logger.info("Loading input dataset")
    data = load_dataset(config.prompts_dataset, revision=dataset_revision)["train"]
    print(data.column_names)
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
    a = 0
    for i, stereotype_dct in enumerate(tqdm(data)):
        logger.info(stereotype_dct)
        index = stereotype_dct["Index"]
        subset = stereotype_dct["Subset"]
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
        # Eventually this should also iterate over models, I suppose.
        logprob_dict = {}
        for language in config.languages:
            try:
                biased_sentence = stereotype_dct[language + ": Biased Sentences"]
                biased_template = stereotype_dct[language + ": Templates"]
                is_expression = stereotype_dct[language + ": Is this a saying?"]
                comments = stereotype_dct[language + ": Comments"]
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
            tokens_key = get_complex_key('tokens', model_name)
            logprob_key = get_complex_key('logprob', model_name)
            logprob_dict[config.language_codes[language]] = {
                logprob_key: logprob,
                tokens_key: tokens,
                'biased_sentence':biased_sentence,
                'biased_template':biased_template,
                'is_expression': is_expression,
                'comments': comments}
        eval_dict = create_dict(index, subset, bias_type, orig_languages, lang_validity, region_validity, stereotyped_group, logprob_dict)
        print(eval_dict)
        eval_dataset['test'] = eval_dataset['test'].add_item(eval_dict)
        a += 1
        print(eval_dataset)
        if a % 30 == 0:
            eval_dataset.push_to_hub('LanguageShades/BiasShadesBaseEval', private=False, token=os.environ.get("HF_TOKEN", None))
    eval_dataset.push_to_hub('LanguageShades/BiasShadesBaseEval', private=False, token=os.environ.get("HF_TOKEN", None))


if __name__ == "__main__":
    main(model_name="bigscience/bloom-7b1")
