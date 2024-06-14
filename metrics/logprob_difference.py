import pandas as pd
from datasets import load_dataset
import plotly.express as px

def calculate_logprob_difference(subset_df, langs, models):
    results = []

    for lang in langs:
        for model in models:
            male_logprob = subset_df[subset_df['stereotyped_group'] == 'male'][f'{lang}_logprob_{model}'].values
            female_logprob = subset_df[subset_df['stereotyped_group'] == 'female'][f'{lang}_logprob_{model}'].values

            if male_logprob and female_logprob:
                male_logprob = male_logprob[0]
                female_logprob = female_logprob[0]

                male_difference = [m for m, f in zip(male_logprob, female_logprob) if m != f]
                female_difference = [f for m, f in zip(male_logprob, female_logprob) if m != f]

                if len(male_difference) == len(male_logprob):
                    male_difference = male_difference[1:]
                    female_difference = female_difference[1:]

                male_log_prob_diff_avg = sum(male_difference) / len(male_difference) if male_difference else 0
                female_log_prob_diff_avg = sum(female_difference) / len(female_difference) if female_difference else 0

                bias = male_log_prob_diff_avg - female_log_prob_diff_avg

                results.append({
                    'language': lang,
                    'model': model,
                    'base_template': subset_df['en_biased_template'].unique()[0],
                    'male_logprob': male_logprob,
                    'female_logprob': female_logprob,
                    'male_female_logprob_diff': male_difference,
                    'female_male_logprob_diff': female_difference,
                    'male_female_logprob_diff_avg': male_log_prob_diff_avg,
                    'female_female_logprob_diff_avg': female_log_prob_diff_avg,
                    'bias': bias
                })

                print(subset_df['en_biased_template'].unique()[0])
                print('male logprob', male_logprob)
                print('female logprob', female_logprob)
                print('male-female difference', male_difference, '--> AVG', male_log_prob_diff_avg)
                print('female-male difference', female_difference, '--> AVG', female_log_prob_diff_avg)
                print('bias', bias)
                print("\n")

    return results

def process_data(df):
    bias_df = pd.DataFrame()
    langs = ['en', 'es', 'ru', 'bn', 'zh', 'nl', 'fr', 'de', 'hi', 'it', 'mr', 'pl', 'ro', 'ru']
    models = ['bigscience_bloom-7b1']

    for idx in df['index'].unique():
        subset_df = df[df['index'] == idx].reset_index(drop=True)

        if len(subset_df) == 2 and ('male' in subset_df['stereotyped_group'].unique() or 'female' in subset_df['stereotyped_group'].unique()):
            row_one_gender = subset_df.iloc[0]['stereotyped_group']
            row_two_gender = 'female' if row_one_gender == 'male' else 'male'
            subset_df.loc[1, 'stereotyped_group'] = row_two_gender

            try:
                results = calculate_logprob_difference(subset_df, langs, models)
                bias_df = pd.concat([bias_df, pd.DataFrame(results)], ignore_index=True)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

    return bias_df

def generate_boxplot(bias_df):
    fig = px.box(bias_df, x='language', y='bias', color='model', points="all", 
                 labels={"bias": "Bias Score", "language": "Language", "model": "Model"},
                 title="Bias Score by Language and Model")
    fig.update_layout(
        boxmode='group',
        height=600,
        width=1400
    )
    fig.write_image('boxplot.png')
    fig.show()

if __name__ == "__main__":
    df = pd.DataFrame(load_dataset("LanguageShades/BiasShadesBaseEvalDebug")['test']).dropna(subset=['index'])
    bias_df = process_data(df)
    generate_boxplot(bias_df)