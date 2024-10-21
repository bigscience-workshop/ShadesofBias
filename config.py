import os


def get_token_path():
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None and os.path.exists(token_path):
        with open(token_path, "r") as file:
            hf_token = file.readline().strip()
    return hf_token


class Config:
    prompts_dataset_raw = "LanguageShades/BiasShadesRaw"
    prompts_dataset_formatted = "LanguageShades/FormattedBiasShades"
    prompts_dataset_logprobs = "LanguageShades/FormattedBiasShadesWithLogprobs"
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
    language_codes = {
        "Arabic": "ar",
        "Bengali": "bn",
        "Brazilian Portuguese": "pt-BR",
        "Dominican Republic Spanish": "es-DO",
        "Chinese": "zh",
        "Traditional Chinese": "zh-hant",
        "Dutch": "nl",
        "English": "en",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Italian": "it",
        "Marathi": "mr",
        "Polish": "pl",
        "Romanian": "ro",
        "Russian": "ru",
        "Uzbekistan Russian": "ru-UZ",
        "Spanish": "es",
    }
    # Languages that are also represented in the stereotypes.
    language_code_list = ["ar", "bn", "pt-BR", "zh", "zh-hant", "nl", "en", "fr", "de", "hi", "it", "mr", "pl", "ro", "ru", "es-DO", "es", "ru-UZ"]
    country_iso_map = {
        "Algeria": "DZA",
        "Bahrain": "BHR",
        "Egypt": "EGY",
        "Iraq": "IRQ",
        "Jordan": "JOR",
        "Kuwait": "KWT",
        "Libya": "LBY",
        "Mauritania": "MRT",
        "Morocco": "MAR",
        "Oman": "OMN",
        "Palestine": "PSE",
        "Qatar": "QAT",
        "Saudi Arabia": "SAU",
        "Sudan": "SDN",
        "Syria": "SYR",
        "Tunisia": "TUN",
        "United Arab Emirates": "ARE",
        "Yemen": "YEM",
        "Mainland China": "CHN",
        "India": "IND",
        "Brazil": "BRA",
        "Uzbekistan": "UZB",
        "Dominican Republic": "DOM",
        "Romania": "ROU",
        "Russia": "RUS",
        "Hong Kong": "HKG",
        "France": "FRA",
        "Netherlands": "NLD",
        "Flemish Belgium": "BEL",  # Assuming it's referring to Belgium
        "Flanders Belgium": "BEL",  # Assuming it's referring to Belgium
        "Poland": "POL",
        "Italy": "ITA",
        "Japan": "JPN",
        "West Germany": "DEU",  # West Germany is now part of Germany (DE)
        "China": "CHN",
        "Germany": "DEU",
        "mainland China": "CHN",
        "Lebanon": "LBN",
        "US": "USA",
        "UK": "GBR"
    }
    basic_cols = ['index', 'subset', 'bias_type', 'stereotype_origin_langs',
                  'stereotype_valid_langs', 'stereotype_valid_regions',
                  'stereotyped_entity', 'type']
    formatted_dataset_columns = ['index', 'subset', 'bias_type', 'stereotype_origin_langs', 'stereotype_valid_langs', 'stereotype_valid_regions', 'stereotyped_entity', 'type', 'en_templates', 'en_biased_sentences', 'en_expression', 'en_comments', 'fr_templates', 'fr_biased_sentences', 'fr_expression', 'fr_comments', 'ro_templates', 'ro_biased_sentences', 'ro_expression', 'ro_comments', 'ar_templates', 'ar_biased_sentences', 'ar_comments', 'ar_expression', 'bn_templates', 'bn_biased_sentences', 'bn_comments', 'bn_expression', 'zh_templates', 'zh_biased_sentences', 'zh_expression', 'zh_comments', 'zh_hant_templates', 'zh_hant_biased_sentences', 'zh_hant_expression', 'zh_hant_comments', 'nl_templates', 'nl_biased_sentences', 'nl_expression', 'nl_comments', 'hi_templates', 'hi_biased_sentences', 'hi_expression', 'hi_comments', 'mr_templates', 'mr_biased_sentences', 'mr_expression', 'mr_comments', 'ru_templates', 'ru_biased_sentences', 'ru_comments', 'ru_expression', 'de_templates', 'de_biased_sentences', 'de_comments', 'de_expression', 'it_templates', 'it_biased_sentences', 'it_expression', 'it_comments', 'pl_templates', 'pl_biased_sentences', 'pl_comments', 'pl_expression', 'pt_br_templates', 'pt_br_biased_sentences', 'pt_br_comments', 'pt_br_expression', 'es_templates', 'es_biased_sentences', 'es_comments', 'es_expression',]

    all_types = ["obligation", "declaration", "aspiration", "conversational", "description", "question"]
    hf_token = os.getenv("HF_TOKEN", get_token_path())

    base_model_list = [
        "Qwen/Qwen2-1.5B"
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-72B",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",
        "bigscience/bloom-7b1",
        "bigscience/bloom-1b7",
        #"bigscience/bloom", # This one is way too big to use, heh.
        "mistralai/Mistral-7B-v0.1",
    ]
