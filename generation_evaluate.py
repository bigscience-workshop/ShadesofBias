import json
import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv

from metrics.metrics import accuracy_aggregate, accuracy_by_lang

load_dotenv()
from loguru import logger
from tqdm import tqdm

from config import Config
from huggingface_api import HFEndpointAPI
from utils import (
    format_single_prompt,
    get_set,
    helper_parse_for_labels,
    load_endpoint_url,
)


def main(
    model_name,
    max_new_tokens=20,
    repetition_penalty=1.0,
    dataset_revision=None,
    api_url=None,
    promptparams={"prompt_name": "reflect"},
):
    config = Config()
    logger.info("Loading dataset")
    data = load_dataset(config.prompts_dataset, revision=dataset_revision)["test"]
    #data = data.shuffle(seed=42).select(range(10))
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

    suffix = ''

    preds = dict()
    preds_raw = dict()
    for language in config.languages:
        preds[f"pred_label{language}"] = [-1] * len(data)
        preds_raw[f"pred_{language}"] = ['null'] * len(data)

    def update_with_labels(example):
        if example["index"] is not None:
            if example["subset"] == '_original' or example["subset"] is None:
                example["gold"] = 1
            else:
                # If it's a contrastive instance, then it is
                # only an overgeneralization instead of stereotype
                example["gold"] = 0
        else:
            example["gold"] = 0
        return example


    data = data.map(update_with_labels)

    for i, stereotype_dct in enumerate(tqdm(data)):
        logger.info(stereotype_dct)
        id = stereotype_dct["index"]
        bias_type = stereotype_dct["bias_type"]
        orig_languages = get_set(stereotype_dct["stereotype_origin_langs"])
        lang_validity = get_set(
            stereotype_dct[
                "stereotype_valid_langs"
            ]
        )
        region_validity = get_set(
            stereotype_dct[
                "stereotype_valid_regions"
            ]
        )
        stereotyped_group = stereotype_dct["stereotyped_entity"]
        for language in config.languages:
            try:
                biased_sentence = stereotype_dct[config.language_codes[language] + "_biased_sentences"]
                if biased_sentence:
                    # Prompt formatting
                    prompt = format_single_prompt(biased_sentence, promptparams, config.language_codes[language])
                    generated_text, success = model_api.query_model(
                        prompt, pred_method="rawgen"
                    )
                else:
                    continue
            except KeyError:
                sys.stderr.write("Fix %s\n" % language)
                continue
            # Parse for lables in generated text
            pred_label = helper_parse_for_labels(
                prompt, generated_text, promptparams['prompt_name']
            )
            logger.info(f"Predicted Label: {pred_label}")
            preds[f"pred_label{language}"][i] = pred_label
            preds_raw[f"pred_{language}"][i] = generated_text
    

    for language in config.languages:
        data = data.add_column(f"{language}: Pred label", preds[f"pred_label{language}"])
        data = data.add_column(f"{language}: Pred output", preds_raw[f"pred_{language}"])
    # Save the final data to preds/
    metrics = {
        "aggregate_acc": accuracy_aggregate(data=data, preds=preds),
    }
    for language in config.languages:
        try:
            metrics[language + "_acc"] = accuracy_by_lang(
                data, preds=preds[f"pred_label{language}"]
            )
        except:
            logger.error(f"No pred for {language}")
    
    model_to_save = model_name.split('/')[1]
    with open(f"preds/metrics_{suffix}{model_to_save}{promptparams['prompt_name']}.json", "w+") as outfile:
        json.dump(metrics, outfile)

    with open(f"preds/gen_predictions/{suffix}{model_to_save}{promptparams['prompt_name']}.json", "w") as outfile:
        json.dump(preds, outfile)
    with open(f"preds/gen_predictions/raw_predictions_{suffix}{model_to_save}{promptparams['prompt_name']}.json", "w") as outfile:
        json.dump(preds, outfile)
    
    data.save_to_disk(f"preds/pred_generate_{suffix}{model_to_save}{promptparams['prompt_name']}")


if __name__ == "__main__":
    main(
        model_name="Qwen/Qwen2-7B-Instruct",
        api_url="https://rnc2dsweb7fdjlpy.us-east-1.aws.endpoints.huggingface.cloud",
        dataset_revision="000d61d",
        promptparams={"prompt_name": "final_prompt2"},
    )