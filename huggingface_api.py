import requests
from loguru import logger
from transformers import AutoTokenizer


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

    def endpoint_generate(self, prompt, constraint=None, append_bos=False):
        if self.tokenizer.bos_token:
            bos_token = self.tokenizer.bos_token
        else:
            bos_token = ''
        if append_bos:
            prompt = bos_token + prompt
        if constraint is not None:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1,
                    "min_new_tokens": 1,
                    "decoder_input_details": True,
                    "grammar": {"type": "regex", "value": r"(Y|N)"},
                },
            }
        else:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.answer_tokens,
                    "repetition_penalty": self.repetition_penalty,
                    "decoder_input_details": True,
                },
            }

        q, success = self.query(payload=payload)
        return q, success

    def query_model(self, prompt, pred_method="logprob", append_bos=False):
        # Remember there will be an error if the model hasn't been pinged in the
        # past hour -- it goes to sleep, then will take a few minutes after you
        # ping it again to wake up.
        try:
            response, success = self.endpoint_generate(prompt, append_bos=append_bos)
        except OSError as e:
            response = e
            success = False
        if success:
            if pred_method == "logprob":
                # Log Prob Computation
                logprobs_prompt, log_probs_answer = (
                    response[0]["details"]["prefill"],
                    response[0]["details"]["tokens"],
                )
                return logprobs_prompt, log_probs_answer, success
            elif pred_method == "rawgen":
                # Generation Computation
                return response[0]["generated_text"], success
                # Return response
            else:
                logger.error(
                    "===== !! Prediction method not supported. Please input from \{logprob, rawgen\}"
                )
        logger.error("===== !! Failed to get model response")
        logger.debug(response)
        return response, None, success
