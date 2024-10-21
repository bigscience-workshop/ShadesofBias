import pandas as pd
from datasets import load_dataset
import plotly.express as px
import sys
import pickle
from config import Config

GROUP1_STR = 'females'
GROUP2_STR = 'males'

def calculate_logprob_difference(subset_df, langs, models, experiment=None):
    results = []

    # EXPERIMENT: Limit just to languages where it is valid.
    stereotype_valid_langs = subset_df['stereotype_valid_langs'].values[0]
    for lang in langs:
        # Experiment option: Don't consider cases where the stereotype isn't
        # valid in that language.
        if experiment == 'stereotype_valid_langs' and lang not in stereotype_valid_langs:
            continue
        for model in models:
            group1_logprob = subset_df[subset_df['subset'] == '_original'][f'{lang}_logprob_{model}'].values
            group2_logprob = subset_df[subset_df['subset'] == 'a'][f'{lang}_logprob_{model}'].values

            if group2_logprob and group1_logprob:
                group1_logprob = group1_logprob[0]
                group2_logprob = group2_logprob[0]

                group1_difference = [f for m, f in zip(group2_logprob, group1_logprob) if m != f]
                group2_difference = [m for m, f in zip(group2_logprob, group1_logprob) if m != f]

                if len(group2_difference) == len(group2_logprob):
                    group2_difference = group2_difference[1:]
                    group1_difference = group1_difference[1:]

                group2_log_prob_diff_avg = sum(group2_difference) / len(group2_difference) if group2_difference else 0
                group1_log_prob_diff_avg = sum(group1_difference) / len(group1_difference) if group1_difference else 0

                bias = group2_log_prob_diff_avg - group1_log_prob_diff_avg

                group1 = subset_df.loc[0, 'stereotyped_entity']
                group2 = subset_df.loc[1, 'stereotyped_entity']
                results.append({
                    'language': lang,
                    'model': model,
                    'original_stereotype': subset_df[lang + '_biased_sentences'].unique()[0],
                    'base_template': subset_df[lang + '_templates'].unique()[0],
                    group2 + '_logprob': group2_logprob,
                    group1 + '_logprob': group1_logprob,
                    group2 + '_' + group1 + '_logprob_diff': group2_difference,
                    group1 + '_' + group2 + '_logprob_diff': group1_difference,
                    group2 + '_' + group1 + '_logprob_diff_avg': group2_log_prob_diff_avg,
                    group1 + '_' + group2 + '_logprob_diff_avg': group1_log_prob_diff_avg,
                    'bias': bias
                })

                print('Original stereotype: %s' % subset_df[lang + '_biased_sentences'].unique()[0])
                print('Template: %s' % subset_df[lang + '_templates'].unique()[0])
                print(group2 + ' logprob', group2_logprob)
                print(group1 + ' logprob', group1_logprob)
                print(group2 + '-' + group1 + ' difference', group2_difference, '--> AVG', group2_log_prob_diff_avg)
                print(group1 + '-' + group2 + ' difference', group1_difference, '--> AVG', group1_log_prob_diff_avg)
                print('bias', bias)
                if group1_log_prob_diff_avg > group2_log_prob_diff_avg:
                    print('bias towards %s for this statement.' % group1)
                else:
                    print('bias towards %s for this statement.' % group2)
                print("\n")

    return results

def process_data(df):
    langs = ['en', 'fr']#, 'zh'] #config.language_codes.values() #['en', 'es', 'ru', 'bn', 'zh', 'nl', 'fr', 'de', 'hi', 'it', 'mr', 'pl', 'ro', 'ru']
    models = ['Qwen_Qwen2-7B'] #['bigscience_bloom-7b1','meta-llama_Meta-Llama-3-8B','mistralai_Mistral-7B-v0.1','Qwen_Qwen2-7B']
    for model in models:
        for lang in langs:
            for gender_identity in (GROUP1_STR, GROUP2_STR):
                bias_df = pd.DataFrame()
                # Get all the stereotypes for a given index.
                for idx in df['index'].unique():
                    subset_df = df[df['index'] == idx].reset_index(drop=True)
                    # Handling for the fact that it won't just be a single identity term
                    # in the original stereotype cell.
                    print(subset_df)
                    print(subset_df['stereotyped_entity'])
                    sys.exit()
                    for row in subset_df.iterrows(): #['stereotyped_entity']:
                        print(row)
                        print(row['stereotyped_entity'])
                        sys.exit()
                        identity_terms = row.split()
                        if gender_identity in identity_terms:
                            logprob_column_name = '_'.join([lang, 'logprob', model])
                            logprobs = subset_df[logprob_column_name]
                            print(logprobs)
                    #identity_terms = subset_df['stereotyped_entity'][0]#.split()
                    #print(identity_terms)
                    sys.exit()
                    orig_identity = subset_df[subset_df['subset'] == '_original']['stereotyped_entity'].unique()[0]
                    orig_identity_terms = orig_identity.split() #[term.strip(',') for term in subset_df[subset_df['subset'] == '_original']['stereotyped_entity'].unique()[0].split()]
                    #except IndexError:
                    #    #print(subset_df[subset_df['subset'] == None])
                    #    orig_identity = subset_df['stereotyped_entity'].unique()[0]
                    #    orig_identity_terms = orig_identity.split() #[term.strip(',') for term in subset_df['stereotyped_entity'].unique()[0].split()]
                    row_one_gender = ''
                    if gender_identity in orig_identity_terms:
                        row_one_gender = gender_identity
        #            row_one_gender = ''
        #            for term in orig_identity_terms:
        #                # Don't deal with multiple entities for this evaluation.
        #                if re.match('and', term) or re.match('&', term):
        #                    break
        #                # Matches the whole string, so eg, 'females' won't be returned
        #                # for a 'males' search. I think.
        #                if re.match(gender_identity, term):
        #                    row_one_gender = gender_identity
                    # Just look at those stereotypes originally for the selected gender.
                    # TODO: Assumption that only 2 genders are represented.
                    #  Might leave things out, and must be changed for concepts
                    #  other than gender
                    if len(subset_df) == 2 and row_one_gender:
                        row_two_gender = GROUP1_STR if row_one_gender == GROUP2_STR else GROUP2_STR
                        # Just keep it simple for calculation: One identity term
                        subset_df.loc[0, 'stereotyped_entity'] = row_one_gender
                        subset_df.loc[1, 'stereotyped_entity'] = row_two_gender
                        try:
                            results = calculate_logprob_difference(subset_df, langs, models)
                            bias_df = pd.concat([bias_df, pd.DataFrame(results)], ignore_index=True)
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            continue
                gendered_dfs[gender_identity] = bias_df
    return gendered_dfs

def generate_boxplot(bias_df, identity=None):
    print("bias df is")
    print(bias_df)
    with open('bias_df.pickle', 'wb') as f:
        pickle.dump(bias_df, f)
    title = "Bias Score by Language and Model"
    if identity:
        title += ": " + identity + " stereotypes"
    fig = px.box(bias_df, x='language', y='bias', color='model', points="all",
                 labels={"bias": "Bias Score", "language": "Language", "model": "Model"},
                 hover_data=["original_stereotype"],
                 title=title)
    fig.update_layout(
        boxmode='group',
        height=600,
        width=1400
    )
    fig.write_image('boxplot.png')
    fig.show()

if __name__ == "__main__":
    df = pd.DataFrame(load_dataset("LanguageShades/FormattedBiasShadesWithLogprobs")['test'])#.dropna(subset=['index'])
    print(df)
    bias_dict = process_data(df)
    print("bias dict")
    print(bias_dict)
    generate_boxplot(bias_dict[GROUP1_STR], GROUP1_STR)
    if not bias_dict[GROUP2_STR].empty:
        generate_boxplot(bias_dict[GROUP2_STR], GROUP2_STR)
    else:
        print("Empty dataframne from %s" % GROUP2_STR)