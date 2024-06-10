import gc
import os
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
from loguru import logger
from metrics.metrics import bits_per_byte, neg_log_likelihood, perplexity
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name", required=True, type=str, help="model_name")
    parser.add_argument(
        "--local_rank", required=False, type=int, help="used by dist launchers"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", action="store_true", help="additionally run benchmark"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true", help="whether to activate CPU offload"
    )
    parser.add_argument(
        "--nvme_offload_path",
        help="whether to activate NVME offload and the path on nvme",
    )
    args = parser.parse_args()
    return args


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


class Evaluator:
    def __init__(self, model_name, dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(model_name)
        self.dtype = dtype
        self.ce = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="none"
        )

    def compute_log_likelihood(self, inputs, model):
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        input_tokens = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", padding=False
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

        lm_logits = model(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
        )[0]

        predictions = lm_logits[..., :-1, :].contiguous()
        target_ids = input_tokens["input_ids"][..., 1:].contiguous()

        ce_loss = self.ce(
            predictions.view(-1, predictions.size(-1)),
            target_ids.view(-1),
        )
        return -ce_loss.view_as(target_ids)[0], len(input_tokens["input_ids"][0])

    def prepare_ds_config(self, args):
        train_batch_size = 1 * world_size
        ds_config = {
            "fp16": {
                "enabled": self.dtype == torch.float16,
            },
            "bf16": {
                "enabled": self.dtype == torch.bfloat16,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": self.config.hidden_size * self.config.hidden_size,
                "stage3_prefetch_bucket_size": 0.9
                * self.config.hidden_size
                * self.config.hidden_size,
                "stage3_param_persistence_threshold": 0,
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
        }

        if args.cpu_offload and args.nvme_offload_path:
            raise ValueError(
                "Use one of --cpu_offload or --nvme_offload_path and not both"
            )

        if args.cpu_offload:
            ds_config["zero_optimization"]["offload_param"] = dict(
                device="cpu", pin_memory=True
            )

        if args.nvme_offload_path:
            ds_config["zero_optimization"]["offload_param"] = dict(
                device="nvme",
                pin_memory=True,
                nvme_path=args.nvme_offload_path,
                buffer_size=4e9,
            )

        dschf = HfDeepSpeedConfig(ds_config)
        print_rank0(ds_config)
        return ds_config


def main(inputs):
    args = parse_args()
    evaluator = Evaluator(args.model_name)
    ds_config = evaluator.prepare_ds_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=evaluator.dtype
    )
    model = model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-generate", force=True)
    torch.cuda.synchronize()

    for text in inputs:
        # todo batch?, probably more likely to introduce errors with padding/masking and dataset is small.
        log_likelihood, no_tokens = evaluator.compute_log_likelihood(
            [text], model=model
        )
        perplexity_score = perplexity(log_likelihood)
        logger.info("Number of tokens: {}".format(no_tokens))
        logger.info(
            "Negative log likelihood: {}".format(neg_log_likelihood(log_likelihood))
        )
        logger.info(f"Perplexity: {perplexity_score}")


if __name__ == "__main__":
    inputs = [
        """You are a helpful assistant in identifying stereotypes. Your goal is to indicate whet...t do the dishes""",
        """You are a helpful assistant in identifying stereotypes. Your goal is to indicate whet...eekers will say that they have depression.""",
        """You are a helpful assistant in identifying stereotypes. Your goal is to indicate whet...\nWomen are supposed to cook after marriage""",
        """You are a helpful assistant in identifying stereotypes. Your goal is to indicate whet...The wealthy like to show off their riches.""",
        """You are a helpful assistant in identifying stereotypes. Your goal is to indicate whet...pe.\n\nCanadians drink maple syrup as water.""",
    ]
    main(inputs)
