import sys

import pandas as pd
from datasets import load_dataset
import plotly.express as px
import pickle
from config import Config
config = Config()

identity_tuples = [("males", "females")]

def calculate_logprob_difference(group1_str, group2_str, group1_row, group2_row, langs, models, experiment='stereotype_valid_langs'):
    logprob_diff_dict_list = []
    # EXPERIMENT: Limit just to languages where it is valid.
    stereotype_valid_langs = group1_row['stereotype_valid_langs']
    #print("Stereotype valid langs is")
    #print(stereotype_valid_langs)
    for lang in langs:
        original_statement = group1_row[lang + '_biased_sentences']
        if original_statement == None:
            continue
        original_template = group1_row[lang + '_templates']
        contrast_statement = group2_row[lang + '_biased_sentences']
        if contrast_statement == None:
            continue
        # Will almost always be the same as the original.
        contrast_template = group2_row[lang + '_templates']

        #print('Original stereotype: %s' % original_statement)
        #print('Template: %s' % original_template)
        # Experiment option: Don't consider cases where the stereotype isn't
        # valid in that language.
        if experiment == 'stereotype_valid_langs' and lang not in stereotype_valid_langs:
            continue
        for model in models:
            ppl_column_name = f'{lang}_ppl_{model}'
            ppl_diff = group1_row[ppl_column_name] - group2_row[ppl_column_name]
            logprob_column_name = f'{lang}_logprob_{model}'
            group1_logprob = group1_row[logprob_column_name]
            group2_logprob = group2_row[logprob_column_name]
            prob_length = len(group1_logprob)
            if len(group1_logprob) > len(group2_logprob):
                prob_length = len(group2_logprob)
            for j in range(prob_length):
                prob1 = group1_logprob[j]
                prob2 = group2_logprob[j]
                if prob1 == prob2:
                    continue
                else:
                    break
            group1_logprob_short = group1_logprob[j:]
            group2_logprob_short = group2_logprob[j:]


            group1_avg_logprob = sum(group1_logprob_short)/len(group1_logprob_short)
            group2_avg_logprob = sum(group2_logprob_short)/len(group2_logprob_short)
            #print(group1 + ' logprob', group1_logprob)
            #print(group2 + ' logprob', group2_logprob)
            bias = group1_avg_logprob - group2_avg_logprob
            #print('bias', bias)
            #if bias > 0:
            #    print("Bias towards %s" % group1)
            #elif bias < 0:
            #    print("Bias towards %s" % group2)
            #else:
            #    print("No bias.")
            #print("\n")

            logprob_diff_dict_list.append({
                    'language': lang,
                    'model': model,
                    'stereotyped_entity': group1_str,
                    'original_stereotype': original_statement,
                    'original_template': original_template,
                    'contrast_stereotype': contrast_statement,
                    'contrast_template': contrast_template,
                    group1_str + '_logprob': group1_logprob,
                    group2_str + '_logprob': group2_logprob,
                    'bias': bias,
                    'ppl_diff': ppl_diff,
                })

    return logprob_diff_dict_list

def _get_row(d, i):
    row_dict = {}
    for key in d:
        value = d[key][i]
        row_dict[key] = value
    return row_dict

def process_data(d, languages, model_list, group1_str, group2_str):
    results = []
    for i in range(len(d['index'])):
        try:
            stereotype_idx = d['index'][i].strip()
            subset = d['subset'][i].strip()
        except:
            continue
        if subset != "_original":
            continue
        try:
            stereotyped_entity_list = d['stereotyped_entity'][i].split()
        except Exception as e:
            print(e)
            print("Regarding stereotyped entities on row %d, index %s" % (i, stereotype_idx))
            continue # missing the stereotyped entities
        # If this row concerns an entity we're focusing on....
        # And isn't using both
        if group1_str in stereotyped_entity_list and group2_str not in stereotyped_entity_list:
            #group1_logprobs = d[logprob_column_name][i]
            group1_row = _get_row(d, i)
            #print("group 1 row is")
            #print(group1_row)
            j = i + 1
            contrast_idx = d['index'][j].strip()
            if contrast_idx == stereotype_idx:
                contrast_entity_list = d['stereotyped_entity'][j].split()
                # TODO: Continue on to the next row if not?
                if group2_str in contrast_entity_list and group1_str not in contrast_entity_list:
                    group2_row = _get_row(d, j)
                    #print(group1_row)
                    #print(group2_row)
                    logprob_diff_dict = calculate_logprob_difference(group1_str, group2_str, group1_row, group2_row, languages, model_list)
                    #print(logprob_diff_dict)
                    results += logprob_diff_dict
                    #print(results)
    return results

def generate_boxplot(bias_df, identity=None):
    title = "Bias Score by Language and Model"
    if identity:
        title += ": " + identity + " stereotypes"
    fid = '_'.join(title.lower().split())
    with open(fid + '.pickle', 'wb') as f:
        pickle.dump(bias_df, f)
    fig = px.box(bias_df, x='language', y='bias', color='model', points="all",
                 labels={"bias": "Bias Score", "language": "Language", "model": "Model"},
                 hover_data=["original_stereotype"],
                 title=title)
    fig.update_layout(
        boxmode='group',
        height=600,
        width=1400
    )
    fid = '_'.join(title.lower().split())
    fig.write_html(fid + '.html')
    fig.write_image(fid + '.png')
    fig.show()

if __name__ == "__main__":
    # What languages are we evaluating?
    languages = config.language_codes.values() #["en", "fr"] #["English", "French"]
    # What models are we evaluating?
    model_list = ["Qwen_Qwen2-7B", "meta-llama_Meta-Llama-3-8B",] # "bigscience/bloom-7b1"]
    d = load_dataset("LanguageShades/FormattedBiasShadesWithLogprobs")['test'].to_dict()
    print(d.keys())
    for group_tuple in identity_tuples:
        group1_str, group2_str = group_tuple
        # First part of tuple as _original
        results = process_data(d, languages, model_list, group1_str=group1_str, group2_str=group2_str)
        bias_df = pd.DataFrame(results)
        generate_boxplot(bias_df, group1_str)


        results2 = process_data(d, languages, model_list, group1_str=group2_str, group2_str=group1_str)
        #print(results2)
        bias_df2 = pd.DataFrame(results2)
        generate_boxplot(bias_df2, group2_str)

        combined_results = results + results2
        combined_df = pd.DataFrame(combined_results)
        generate_boxplot(combined_results, "male and female")
        # Second part of the tuple as _original
        #bias_df = process_data(d, languages, model_list, group1_str=group2_str, group2_str=group1_str)
        #generate_boxplot(bias_df, group2_str)

        # Second identity in tuple as _original
    #if not bias_dict[GROUP2_STR].empty:
    #    generate_boxplot(bias_dict[GROUP2_STR], GROUP2_STR)
    #else:
     #   print("Empty dataframne from %s" % GROUP2_STR)