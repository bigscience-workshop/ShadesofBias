# ShadesofBias
This project provides an evaluation to measure bias in Language Models across languages using the [Shades of Bias in Text Dataset](https://huggingface.co/datasets/LanguageShades/BiasShades).


## HF Endpoints
To use HF Endpoint navigate to [Shades](https://ui.endpoints.huggingface.co/LanguageShades/endpoints) if you have access. If not copy the .env file in your root directory.

## Example Script
Run `example_logprob_evaluate.py` to iterate through the dataset for a given model and compute log probability of biased sentences. If you have the .env, load_endpoint_url(model_name) will load the model if it has been created for that model.
