"""
This file is adapted from https://github.com/allenai/open-instruct/blob/main/eval/predict.py, with support of prediction generation added.
This script is used to get models' predictions on a set of prompts (put in files with *.jsonl format, 
with the prompt in a `prompt` field or the conversation history in a `messages` field).

For example, to get predictions on a set of prompts, you should put them in a file with the following format:
    {"id": <uniq_id>, "prompt": "Plan a trip to Paris."}
    ...
Or you can use the messages format:
    {"id": <uniq_id>, "messages": [{"role": "user", "content": "Plan a trip to Paris."}]}
    ...

"""

import argparse
import json
import os
import pdb

import numpy as np
import torch
from utils import (
    dynamic_import_function,
    generate_completions,
    get_next_word_predictions,
    load_hf_lm_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, help="Huggingface model name or path."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, help="Huggingface tokenizer name or path."
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="Number of shots to provide to the generation.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer.",
    )
    parser.add_argument(
        "--constraint_type",
        type=str,
        help="The type of constraint when predicting classifications, can be {next_word, constraint, non}.",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Input .jsonl files, with each line containing `id` and `prompt` or `messages`.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/model_outputs.jsonl",
        help="Output .jsonl file, with each line containing `id`, `prompt` or `messages`, and `output`.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for prediction."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--load_in_float16",
        action="store_true",
        help="By default, huggingface model will be loaded in the torch.dtype specificed in its model_config file."
        "If specified, the model dtype will be converted to float16 using `model.half()`.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="whether to use sampling ; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="temperature for sampling."
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for sampling.")
    args = parser.parse_args()
    return args


def convert_pred_ids(pred_indices, unconcate_ids, force_words):
    # Map predictions from concated_ids into unconcate_ids
    idx_map = dict()
    concat_idx = 0
    for list_idx, sublist in enumerate(unconcate_ids):
        for _ in sublist:
            idx_map[concat_idx] = list_idx
            concat_idx += 1
    # force word
    pred_texts = [force_words[idx_map[pred]] for pred in pred_indices]
    return pred_texts


if __name__ == "__main__":
    args = parse_args()

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load the data
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            instances = [json.loads(x) for x in f.readlines()]

    # Load chat formatting function
    chat_formatting_function = (
        dynamic_import_function(args.chat_formatting_function)
        if args.use_chat_format
        else None
    )

    if args.num_shots != 0:
        # TODO: Craft a prompt shots
        prefix_prompt = ""
        for instance in instances[0 : args.num_shots]:
            instance_format = ""
            if "messages" in instance:
                prefix_prompt += (
                    instance["messages"][0]["content"]
                    + instance["messages"][1]["content"]
                    + "\n"
                )
            elif "prompt" in instance:
                prefix_prompt += instance["prompt"] + instance["completion"] + "\n"
            else:
                raise ValueError(
                    "Either `messages` or `prompt` should be in the instance."
                )
    else:
        prefix_prompt = ""
    print("Predicting with prefix prompt: ", prefix_prompt)

    if args.model_name_or_path is not None:
        prompts = []
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

        for instance in instances:
            if "messages" in instance:
                if not args.use_chat_format:
                    raise ValueError(
                        "If `messages` is in the instance, `use_chat_format` should be True."
                    )
                assert all(
                    "role" in message and "content" in message
                    for message in instance["messages"]
                ), "Each message should have a `role` and a `content` field."
                prompt = prefix_prompt + chat_formatting_function(
                    instance["messages"], tokenizer=tokenizer, add_bos=False
                )
            elif "prompt" in instance:
                if args.use_chat_format:
                    messages = [{"role": "user", "content": instance["prompt"]}]
                    prompt = prefix_prompt + chat_formatting_function(
                        messages, add_bos=False
                    )
                else:
                    prompt = prefix_prompt + instance["prompt"]
            else:
                raise ValueError(
                    "Either `messages` or `prompt` should be in the instance."
                )
            prompts.append(prompt)
        if args.constraint_type == "non":
            # Constraint can be next_word, constraint, non
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        elif args.constraint_type == "constraint":
            # TODO(Tentative): Generate next_word_prediction
            if "nli" in args.input_files[0] or "rte" in args.input_files[0]:
                force_words = [
                    "entailment",
                    "neutral",
                    "contradiction",
                    " entailment",
                    " neutral",
                    " contradiction",
                ]
                #  ' 0', ' 1', ' 2'
            else:
                # Other tasks
                force_words = ["yes", "no", " yes", " no"]
                # ' 0', ' 1'
            # pass the constraint as args for generate completions
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                force_words_ids=[tokenizer(force_words).input_ids],
                top_p=args.top_p,
                num_beams=5,
            )
        elif args.constraint_type == "next_word":
            if "nli" in args.input_files[0] or "rte" in args.input_files[0]:
                force_words = [
                    "entailment",
                    "neutral",
                    "contradiction",
                    " entailment",
                    " neutral",
                    " contradiction",
                ]
                #  ' 0', ' 1', ' 2'
            else:
                # Paraphrase detection
                force_words = ["yes", "no", " yes", " no"]
            candidate_token_ids = np.concatenate(
                tokenizer(force_words).input_ids
            ).tolist()
            predictions, probs = get_next_word_predictions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                candidate_token_ids=candidate_token_ids,
                batch_size=args.batch_size,
                return_token_predictions=False,
                add_special_tokens=True,
            )
            # Convert prediction indices to prediction ids
            outputs = convert_pred_ids(
                predictions, tokenizer(force_words).input_ids, force_words
            )

        with open(args.output_file, "w") as f:
            for instance, output in zip(instances, outputs):
                instance["output"] = output
                f.write(json.dumps(instance) + "\n")

    else:
        raise ValueError("Model_name_or_path should be provided.")

    print("Done.")
