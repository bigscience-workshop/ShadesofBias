import os
import re


def load_endpoint_url(model_name):
    env_var_key = "MODEL_" + re.sub(r"\W", "_", model_name).upper()
    endpoint_url = os.getenv(env_var_key)
    if endpoint_url:
        return endpoint_url
    else:
        raise ValueError(
            f"Endpoint URL for {model_name} not found in environment variables"
        )


def get_set(entry):
    if entry:
        return [i.strip() for i in entry.split(",")]
