import os


def get_token_path():
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None and os.path.exists(token_path):
        with open(token_path, "r") as file:
            hf_token = file.readline().strip()
    return hf_token


class Config:
    prompts_dataset = "LanguageShades/BiasShades"
    endpoint_api = "https://api-inference.huggingface.co/models/"
    languages = [
        "Arabic",
        "Bengali",
        "Brazilian Portuguese",
        "Chinese",
        "Traditional Chinese",
        "Dutch",
        "English",
        "French",
        "German",
        "Hindi",
        "Italian",
        "Marathi",
        "Polish",
        "Romanian",
        "Russian",
        "Spanish",
    ]
    hf_token = os.getenv("HF_TOKEN", get_token_path())
