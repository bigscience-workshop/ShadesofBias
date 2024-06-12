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
    max_new_tokens=1,
    repetition_penalty=1.0,
    dataset_revision=None,
    api_url=None,
    promptparams={"prompt_lang": "en", "prompt_name": "reflect"},
):
    config = Config()
    logger.info("Loading dataset")
    data = load_dataset(config.prompts_dataset, revision=dataset_revision)["train"]
    data = data.shuffle(seed=42).select(range(2))
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

    preds = dict()
    for language in config.languages:
        preds[f"pred_label{language}"] = [-1] * len(data)

    def update_with_labels(example):
        if example["Index"] is not None and example["Index"].upper().isupper():
            # If it's a contrastive instance, then it is
            # only an overgeneralization instead of stereotype
            example["gold"] = 0
        else:
            example["gold"] = 1
        return example

    data = data.map(update_with_labels)

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
                        prompt, pred_method="rawgen"
                    )
                else:
                    continue
            except KeyError:
                sys.stderr.write("Fix %s\n" % language)
                continue
            # Parse for lables in generated text
            pred_label = helper_parse_for_labels(
                generated_text[len(prompt) :], labels=["no", "yes"]
            )
            logger.info(f"Predicted Label: {pred_label}")
            preds[f"pred_label{language}"][i] = pred_label
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
    with open("preds/metrics.json", "w") as outfile:
        json.dump(metrics, outfile)
    data.save_to_disk(f"preds/pred_generate_{promptparams['prompt_name']}")


if __name__ == "__main__":
    main(
        model_name="bigscience/bloom-7b1",
        dataset_revision="48897fd",
        promptparams={"prompt_lang": "en", "prompt_name": "reflect"},
    )
