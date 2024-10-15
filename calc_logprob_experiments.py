import sys

import pandas as pd
import re
from datasets import load_dataset
import plotly.express as px
import pickle
from config import Config
config = Config()

identity_tuples = [("males", "females")]
global bias_scores

bias_scores = {}
def _get_column_name(model_name, column_type=None):
    """Constructs a string based on the model name and column type.
    This is then used as a dictionary key."""
    flat_model_name = re.sub("/", "_", model_name)
    if column_type:
        key = "_".join([column_type, flat_model_name])
        return key
    return flat_model_name

def _calculate_bias(group1_logprob, group2_logprob):
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
    group1_avg_logprob = sum(group1_logprob_short) / len(group1_logprob_short)
    group2_avg_logprob = sum(group2_logprob_short) / len(group2_logprob_short)
    bias = group1_avg_logprob - group2_avg_logprob
    return bias


def _get_row(d, i):
    row_dict = {}
    for key in d:
        value = d[key][i]
        row_dict[key] = value
    return row_dict

def calculate_logprobs(group1_str, group2_str, group1_row, group2_row, langs, models, experiment='stereotype_valid_langs'):
    global bias_scores
    logprob_diff_dict_list = []
    visualization_dict_list = []
    # EXPERIMENT SETTING: Limit just to languages where it is valid.
    stereotype_valid_langs = group1_row['stereotype_valid_langs']
    index = group1_row['index']
    #print("Stereotype valid langs is")
    #print(stereotype_valid_langs)
    for lang in langs:
        if lang == "es-DO" or lang == "ru-UZ" or lang == "zh_hant":
            continue
        lang = "_".join(lang.lower().split("-"))
        original_statement = group1_row[lang + '_biased_sentences']
        if original_statement == None:
            continue
        is_expression = group1_row[lang + "_expression"]
        is_description = group1_row['bias_type'] == 'description'
        # OPTIONAL
        if is_expression or is_description:
            continue
        original_template = group1_row[lang + '_templates']
        contrast_statement = group2_row[lang + '_biased_sentences']
        if contrast_statement == None:
            continue
        # Will almost always be the same as the original.
        contrast_template = group2_row[lang + '_templates']

        for model_name in models:
            if model_name.startswith("Qwen"):
                model_family = "Qwen"
            elif model_name.startswith("meta"):
                model_family = "Llama"
            else:
                pass
            model = _get_column_name(model_name)
            ppl_column_name = f'{lang}_ppl_{model}'
            try:
                ppl_diff = group1_row[ppl_column_name] - group2_row[ppl_column_name]
            except TypeError:
                continue
            logprob_column_name = f'{lang}_logprob_{model}'
            group1_logprob = group1_row[logprob_column_name]
            group2_logprob = group2_row[logprob_column_name]
            bias = _calculate_bias(group1_logprob, group2_logprob)
            original_ll_avg = sum(group1_logprob)/len(group1_logprob)
            contrast_ll_avg = sum(group2_logprob)/len(group2_logprob)

            # Experiment option: Don't consider cases where the stereotype isn't
            # valid in that language.
            if experiment == 'stereotype_valid_langs' and lang not in stereotype_valid_langs:
                continue
            else:
                logprob_diff_dict_list.append({
                        'language': lang,
                        'model': model,
                        'model_family': model_family,
                        'index': index,
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
                if (model, lang, group1_str) not in bias_scores:
                    bias_scores[(model, lang, group1_str)] = [bias]
                else:
                    bias_scores[(model, lang, group1_str)] += [bias]
            visualization_dict_list.append({
                'index': index,
                'language': lang,
                'model': model,
                'target_entity': group1_str,
                'statement': original_statement,
                'stereotypicality': 'original',
                'stereotypicality x target': 'original x ' + group1_str,
                'model x stereotypicality': model + ' x original',
                'model x stereotypicality x language': model + ' x original x ' + lang,
                'll_avg': original_ll_avg,
            })
            visualization_dict_list.append({
                'index': index,
                'language': lang,
                'model': model,
                'target_entity': group2_str,
                'statement': contrast_statement,
                'stereotypicality': 'contrast',
                'stereotypicality x target': 'contrast x ' + group2_str,
                'model x stereotypicality': model + ' x contrast',
                'model x stereotypicality x language': model + ' x contrast x ' + lang,
                'll_avg': contrast_ll_avg,
            })
    return logprob_diff_dict_list, visualization_dict_list

def process_data(d, languages, model_list, group1_str, group2_str):
    results = []
    visualization_results = []
    for i in range(len(d['index'])):
        #try:
        stereotype_idx = d['index'][i]
        assert isinstance(stereotype_idx, float)
        subset = d['subset'][i].strip()
        #except:
        #    print("fail")
        #    continue
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
            group1_row = _get_row(d, i)
            j = i + 1
            contrast_idx = d['index'][j]#.strip()
            if contrast_idx == stereotype_idx:
                contrast_entity_list = d['stereotyped_entity'][j].split()
                # TODO: Continue on to the next row if not?
                if group2_str in contrast_entity_list and group1_str not in contrast_entity_list:
                    group2_row = _get_row(d, j)
                    logprob_diff_dict_list, visualization_dict_list = calculate_logprobs(
                        group1_str, group2_str, group1_row, group2_row,
                        languages, model_list)
                    results += logprob_diff_dict_list
                    visualization_results += visualization_dict_list
    return results, visualization_results

def generate_distributions(visualization_df, identity=None):
    title = "Average log likelihood"
    if identity:
        title += ": " + identity + " stereotypes"
    fid = '_'.join(title.lower().split())
    with open(fid + '.pickle', 'wb') as f:
        pickle.dump(visualization_df, f)
    # marginal="rug",
    #fig = px.histogram(visualization_df, x="language", y="ll_avg", color="stereotypicality",
    #                   hover_data=visualization_df.columns)
    fig = px.scatter(visualization_df, x='index', y='ll_avg',
                     color='stereotypicality', facet_row='language',
                     labels={'ll_avg': 'Average log-likelihood', 'model': 'Model', 'language': 'Language'},
                     facet_col='model', hover_data=['statement'],
                     title=title)
    fig.show()
    fig.write_html(fid + '.html')
    fig.write_image(fid + '.png')


def generate_bias_boxplot(bias_df, identity=None):
    title = "Bias Score by Language and Model"
    if identity:
        title += ": " + identity + " stereotypes"
    fid = 'log_likelihood_results/' + '_'.join(title.lower().split())
    with open(fid + '.pickle', 'wb') as f:
        pickle.dump(bias_df, f)
    fig = px.box(bias_df, x='language', y='bias', color='model', points="all", facet_col='stereotyped_entity', facet_row='model_family',
                 labels={"bias": "Bias Score", "language": "Language", "model": "Model"},
                 hover_data=["original_stereotype"],
                 title=title)
    fig.update_layout(
        boxmode='group')
    #    height=600,
    #    width=1400
    #)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.6,
        xanchor="right",
        x=0.6
    ))
    fig.update_xaxes(matches=None)
    fig.update_layout(
        font=dict(
            #family="Courier New, monospace",
            size=20,  # Set the font size here
            #color="RebeccaPurple"
        )
    )
    fig.show()
    fig.write_html(fid + '.html')
    fig.write_image(fid + '.png')


def print_averages():
    global key, values, model, lang, group1_str, average
    average_list = []
    for key, values in bias_scores.items():
        model, lang, group1_str = key
        average = sum(values) / float(len(values))
        print("%s %s %s %.2f" % (model, lang, group1_str, average))
        average_list += [(average, model, lang, group1_str)]
    average_list.sort()
    for thing in average_list:
        print(thing)


if __name__ == "__main__":
    # What languages are we evaluating?
    languages = ['zh', 'zh_hant', 'en']
    #languages = ['ar', 'zh', 'nl', 'en', 'fr', 'de', 'it', 'pl', 'ru', 'es']#config.language_code_list #['en', 'zh']#config.language_code_list
    #languages = [Arabic, Bengali, Chinese, Dutch, English, French, German, Hindi, Portuguese, Italian, Polish, Russian, Spanish]
    # What models are we evaluating?
    model_list = ["Qwen/Qwen2-1.5B","Qwen/Qwen2-7B","Qwen/Qwen2-72B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B", ] #["Qwen_Qwen2-7B", "meta-llama_Meta-Llama-3-8B",] # "bigscience/bloom-7b1"]
    d = load_dataset("LanguageShades/FormattedBiasShadesWithLogprobs")['test'].to_dict()
    #print(d.keys())
    for group_tuple in identity_tuples:
        group1_str, group2_str = group_tuple
        # First part of tuple as _original
        results, visualization_results = process_data(d, languages, model_list, group1_str=group1_str, group2_str=group2_str)
        results_df = pd.DataFrame(results)
        #generate_bias_boxplot(results_df, group1_str)
        #visualization_df = pd.DataFrame(visualization_results)
        #generate_distributions(visualization_df, group1_str)

        results2, visualization_results2 = process_data(d, languages, model_list, group1_str=group2_str, group2_str=group1_str)
        #print(results2)
        results_df2 = pd.DataFrame(results2)
        #generate_bias_boxplot(results_df2, group2_str)
        #generate_distributions(visualization_df, group2_str)

        combined_results = results + results2
        combined_df = pd.DataFrame(combined_results)
        generate_bias_boxplot(combined_results, group1_str + " and " + group2_str)

        print_averages()