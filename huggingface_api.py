import requests
from loguru import logger
from transformers import AutoTokenizer

from config import Config


class HFEndpointAPI:
    def __init__(
        self,
        model_name,
        config,
        answer_tokens,
        repetition_penalty,
        api_url=None,
        hf_token=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if api_url is not None:
            self.api_url = api_url
        else:
            self.api_url = config.endpoint_api
        self.hf_token = hf_token if hf_token else config.hf_token
        self.answer_tokens = answer_tokens
        self.repetition_penalty = repetition_penalty

    def query(self, payload):
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json(), response.status_code == 200

    def endpoint_generate(self, prompt):
        q, success = self.query(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.answer_tokens,
                    "repetition_penalty": self.repetition_penalty,
                    "decoder_input_details": True,
                },
            },
        )
        return q, success

    def query_model(self, prompt):
        # Remember there will be an error if the model hasn't been pinged in the
        # past hour -- it goes to sleep, then will take a few minutes after you
        # ping it again to wake up.
        try:
            response, success = self.endpoint_generate(prompt)
        except OSError as e:
            response = e
            success = False
        if success:
            logprobs_prompt, log_probs_answer = (
                response[0]["details"]["prefill"],
                response[0]["details"]["tokens"],
            )
            return logprobs_prompt, log_probs_answer, success
        logger.error("===== !! Failed to get model response")
        logger.debug(response)
        return response, success