# ShadesofBias
This repository provides scripts and code use in the [Shades of Bias in Text Dataset](https://huggingface.co/datasets/LanguageShades/BiasShades).
It includes code for processing the data, and for evaluation to measure bias in Language Models across languages.

## Evaluation

### HF Endpoints
To use HF Endpoint navigate to [Shades](https://ui.endpoints.huggingface.co/LanguageShades/endpoints) if you have access. If not copy the .env file in your root directory.

### Example Script
Run `example_logprob_evaluate.py` to iterate through the dataset for a given model and compute log probability of biased sentences. If you have the .env, load_endpoint_url(model_name) will load the model if it has been created for that model.

Run `generation_evaluate.py` to iterate through the dataset, with each instance formatted with a specified prompt from `prompts/`. It is possible to specify a prompt language that is different from the original language. Prompt language will be set to Enlish unless further specified. If you have the .env, load_endpoint_url(model_name) will load the model if it has been created for that model.

#### Add more prompts
Follow the examples in `prompts/` to create a `.txt` file for new prompt. Input field should be indicated with `{input}` in the text file.

### Base Models
Current [Proposed Model List](https://docs.google.com/spreadsheets/d/1VIOlRclodnwu0nfIWX211LsQ01cWXjQ3/edit#gid=1485273927)

### 'Aligned' models
Todo
