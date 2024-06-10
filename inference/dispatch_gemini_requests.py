import os
from typing import Any, List

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

genai.configure()


def list_available_text_models():
    """
    Returns a list of available generative models for text processing.

    Args:
        token_id(`int`):
            The id of a newly generated token in the beam search.
    Return:
        List of available generative models for text processing
    """
    base_models = [
        m
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
        and "vision" not in m.name
    ]

    return [x.name for x in base_models]


def instantiate_model(model_name: str, **kwargs: Any):
    return genai.GenerativeModel(model_name=model_name, **kwargs)


def query_model(prompt: str, model: genai.GenerativeModel, **kwargs: Any):
    try:
        response = model.generate_content(prompt, **kwargs)
        return response.text
    except Exception as e:
        # print(e)
        return ""


def dispatch_gemini_requesets(
    messages_list: List[str], model: genai.GenerativeModel, **kwargs: Any
) -> List[str]:
    """
    Sends a list of queries to the specified model.

    Args:
        messages_list:
            List of queries to send to the model.
        model:
            A model (genai.GenerativeModel) to query.
    Return:
        List of model responses or empty strings for queries for which an error occurred.
    """
    return [query_model(prompt, model, **kwargs) for prompt in messages_list]


if __name__ == "__main__":
    model_list = list_available_text_models()
    print(f"Avalilable models\n{model_list}")

    model = instantiate_model(
        model_name="models/gemini-1.0-pro",
        generation_config={"temperature": 0.1, "top_k": 1, "top_p": 0.1},
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

    responses = dispatch_gemini_requesets(
        messages_list=[
            "Complete the sentence with a short poem: Generative models are like...",
            "Write a two-line poem about yourself.",
        ],
        model=model,
    )

    for response in responses:
        print(f"{response}\n")
