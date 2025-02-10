from datasets import load_dataset
import pandas as pd
import pdb
from collections import defaultdict as ddict
import argparse
import statsmodels
import numpy as np
import json
from collections import defaultdict as ddict
from statsmodels.formula.api import ols, logit
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.DataFrame(load_dataset("LanguageShades/BiasShadesBaseEval_ALL")['test']).dropna(subset=['index'])
# bias_dict = process_data(df)
# # pdb.set_trace()

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


def create_bias_df():

    ds = load_dataset('LanguageShades/FormattedBiasShadesWithLogprobs')['test']
    # can also call config.language_code_list
    LANGUAGES = ["ar", "bn", "pt_br", "zh", "zh_hant", "nl", "en", "fr", "de", "hi", "it", "mr", "pl", "ro", "ru", "es"]

    original_logprobs_dict = {}

    stereotyped_dict = ddict(list)

    total_cnt = 0
    error_cnt = 0

    with open('bias_type_dict.json', 'r') as f:
        bias_type_dict =  json.load(f)
    

    for model in ['meta-llama_Meta-Llama-3-8B', 'bigscience_bloom-7b1', 'Qwen_Qwen2-7B']:
        # total_cnt = 0
        # error_cnt = 0

        for line in ds:
            # Id for the statement: Both the stereotype and contrasts
            index = line['index']
            # If the value is '_original', then it is the original stereotype.
            # Anything else is a contrast, a non-stereotype.
            subset = line['subset']
            # The type of stereotype it is: age, gender, etc
            bias_type = line['bias_type']

            if bias_type[1:-1] == '': continue
            # The language codes where, in the corresponding regions, the statement might be considered true by some.
            # NOTE: It is useful for some experiments to have statements limited by this, so they are ONLY
            # the ones that are actually observed in the given language/region.
            stereotype_valid_langs = line['stereotype_valid_langs']
            # Which characteristics are being stereotyped in this?
            # NOTE: Multiple characteristics may be listed here.
            # intersections are marked with an intersection symbol, e.g., 'girls' is females ∩ children.
            # multiple entities are separated with 'and'
            stereotyped_entity = line['stereotyped_entity']
            # What kind of stereotype is this?
            type = line['type']
            for lang in LANGUAGES:
                # Is this an expression in the given language region? That will probably affect its likelihood.
                is_expression = line[lang + "_expression"]
                
                total_cnt += 1
                
                try:
                    logprob_column_name = '_'.join([lang, 'logprob', model])
                    logprobs = line[logprob_column_name]

                    if subset == "_original":
                        key = (index, lang, model)
                        original_logprobs_dict[key] = logprobs
                    else:
                        original_logprobs = original_logprobs_dict[key]
                        bias = _calculate_bias(original_logprobs, logprobs)

                        for bt in bias_type_dict[bias_type]:
                            stereotyped_dict['index'].append(index)
                            stereotyped_dict['bias_type'].append(bt)
                            stereotyped_dict['LLM'].append(model)
                            stereotyped_dict['language'].append(lang)
                            stereotyped_dict['bias'].append(bias)

                except Exception as e:
                    error_cnt += 1
                    # print(e)
                    # print("Error at index %s, lang %s, model %s" % (index, lang, model))
                    continue

                

    stereotyped_df = pd.DataFrame(stereotyped_dict)
    stereotyped_df.to_csv('bias_shades_logprob_diff.csv')
    print(f'Error count: {100* error_cnt/total_cnt}')

    print(f'Unique bias types: {stereotyped_df["bias_type"].unique()}')
    print(f'Unique languages: {stereotyped_df["language"].unique()}')

   

def anova_sigtest():

    df          = pd.read_csv('bias_shades_logprob_diff.csv')
    reg_model   = ols('bias ~ C(bias_type) + C(LLM) + C(language)+ C(LLM):C(language)+ C(language):C(bias_type) + C(bias_type):C(LLM) + C(language):C(LLM):C(bias_type)', data=df).fit()
    anova_table = sm.stats.anova_lm(reg_model, typ=2)

    results_text = f'''
    Anova Results:\n
    {anova_table}\n
    ====================================================================================\n
    Regression Model Parameters:\n
    {reg_model.params}\n
    '''

    df          = pd.read_csv('bias_shades_logprob_diff.csv')
    reg_model2   = ols('bias ~ C(bias_type) + C(LLM) + C(language)+ C(LLM):C(language)+ C(language):C(bias_type) + C(bias_type):C(LLM)', data=df).fit()
    anova_table2 = sm.stats.anova_lm(reg_model2, typ=2)

    results_text += f'''
    Anova Results:\n
    {anova_table2}\n
    ====================================================================================\n
    Regression Model Parameters:\n
    {reg_model2.params}\n
    '''

    diff_results = sm.stats.anova_lm(reg_model, reg_model2)
    
    results_text += f'''

    Difference in Anova Results:\n
    {diff_results}\n
    '''

    with open('anova_results.txt', 'w') as f:
        f.write(results_text)


def kstest():
    df = pd.read_csv('bias_shades_logprob_diff.csv')

    LANGUAGES   = ["ar", "bn", "pt_br", "zh", "zh_hant", "nl", "en", "fr", "de", "hi", "it", "mr", "pl", "ro", "ru", "es"]
    LLMS        = ['meta-llama_Meta-Llama-3-8B', 'bigscience_bloom-7b1', 'Qwen_Qwen2-7B']
    BIAS_TYPES  = list(df['bias_type'].unique())
    
    print(f'Bias types: {BIAS_TYPES}')
    # if comparing between languages you need to ensure that they are from the same model and bias type

    # for model in MODELS:
        # for bias in BIAS_TYPES:

    ks_results = ddict(list)

    total_cnt = 0
    error_cnt = 0

    for lang1 in LANGUAGES:
        for lang2 in LANGUAGES:

            if lang1 == lang2:
                continue
            
            df1 = df[(df['language'] == lang1)]
            df2 = df[(df['language'] == lang2)]

            new_df = df1.merge(df2, on=['index', 'LLM'], suffixes=('_1', '_2'))

            bias1 = list(new_df['bias_1'].values)
            bias2 = list(new_df['bias_2'].values)

            total_cnt += 1

            print(f'Done for {total_cnt} samples', end='\r')


            try:
                statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                ks_results['dimension'].append('language')
                ks_results['control'].append('No')
                ks_results['case1'].append(lang1)
                ks_results['case2'].append(lang2)
                ks_results['#samples'].append(len(bias1))

                ks_results['mean_bias_1'].append(np.mean(bias1))
                ks_results['mean_bias_2'].append(np.mean(bias2))
                ks_results['statistic'].append(float(statistic))
                ks_results['pval'].append(float(pval))


            except Exception as e:
                error_cnt += 1
                continue


    for LLM1 in LLMS:
        for LLM2 in LLMS:
            if LLM1 == LLM2:
                continue
            
            df1 = df[(df['LLM'] == LLM1)]
            df2 = df[(df['LLM'] == LLM2)]

            new_df = df1.merge(df2, on=['index', 'language'], suffixes=('_1', '_2'))

            bias1 = list(new_df['bias_1'].values)
            bias2 = list(new_df['bias_2'].values)

            total_cnt += 1

            print(f'Done for {total_cnt} samples', end='\r')

            try:
                statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                ks_results['dimension'].append('LLM')
                ks_results['control'].append('No')
                ks_results['case1'].append(LLM1)
                ks_results['case2'].append(LLM2)
                ks_results['#samples'].append(len(bias1))
                ks_results['mean_bias_1'].append(np.mean(bias1))
                ks_results['mean_bias_2'].append(np.mean(bias2))
                ks_results['statistic'].append(float(statistic))
                ks_results['pval'].append(float(pval))
            except Exception as e:
                error_cnt += 1
                continue


    for bt1 in BIAS_TYPES:
        for bt2 in BIAS_TYPES:
            if bt1 == bt2:
                continue
            

            df1 = df[(df['bias_type'] == bt1)]
            df2 = df[(df['bias_type'] == bt2)]

            new_df = df1.merge(df2, on=['LLM', 'language'], suffixes=('_1', '_2'))

            bias1 = list(new_df['bias_1'].values)
            bias2 = list(new_df['bias_2'].values)

            total_cnt += 1

            print(f'Done for {total_cnt} samples', end='\r')
            try:
                statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                ks_results['dimension'].append('bias_type')
                ks_results['control'].append('No')
                ks_results['case1'].append(bt1)
                ks_results['case2'].append(bt2)
                ks_results['#samples'].append(len(bias1))
                ks_results['mean_bias_1'].append(np.mean(bias1))
                ks_results['mean_bias_2'].append(np.mean(bias2))
                
                ks_results['statistic'].append(float(statistic))
                ks_results['pval'].append(float(pval))
            except Exception as e:
                print(e)
                error_cnt += 1
                continue

    
    # control for the LLM while measuring the difference between languages
    for model in LLMS:
        for lang1 in LANGUAGES:
            for lang2 in LANGUAGES:

                if lang1 == lang2:
                    continue
                
                df1 = df[(df['language'] == lang1)& (df['LLM'] == model)]
                df2 = df[(df['language'] == lang2)& (df['LLM'] == model)]

                new_df = df1.merge(df2, on=['index'], suffixes=('_1', '_2'))

                bias1 = list(new_df['bias_1'].values)
                bias2 = list(new_df['bias_2'].values)

                total_cnt += 1
                print(f'Done for {total_cnt} samples', end='\r')
                try:
                    statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                    ks_results['dimension'].append('language')
                    ks_results['control'].append(f'LLM: {model}')
                    ks_results['case1'].append(lang1)
                    ks_results['case2'].append(lang2)
                    ks_results['#samples'].append(len(bias1))

                    ks_results['mean_bias_1'].append(np.mean(bias1))
                    ks_results['mean_bias_2'].append(np.mean(bias2))
                    ks_results['statistic'].append(float(statistic))
                    ks_results['pval'].append(float(pval))


                except Exception as e:
                    error_cnt += 1
                    continue

    # control for the language while measuring the difference between LLMs
    for lang in LANGUAGES:
        for LLM1 in LLMS:
            for LLM2 in LLMS:
                if LLM1 == LLM2:
                    continue
                
                df1 = df[(df['LLM'] == LLM1)& (df['language'] == lang)]
                df2 = df[(df['LLM'] == LLM2)& (df['language'] == lang)]

                new_df = df1.merge(df2, on=['index'], suffixes=('_1', '_2'))

                bias1 = list(new_df['bias_1'].values)
                bias2 = list(new_df['bias_2'].values)

                print(f'Done for {total_cnt} samples', end='\r')
                total_cnt += 1
                try:
                    statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                    ks_results['dimension'].append('LLM')
                    ks_results['control'].append(f'language: {lang}')
                    ks_results['case1'].append(LLM1)
                    ks_results['case2'].append(LLM2)
                    ks_results['#samples'].append(len(bias1))
                    ks_results['mean_bias_1'].append(np.mean(bias1))
                    ks_results['mean_bias_2'].append(np.mean(bias2))
                    ks_results['statistic'].append(float(statistic))
                    ks_results['pval'].append(float(pval))
                except Exception as e:
                    error_cnt += 1
                    continue

    # control for all parameters except one

    for lang in LANGUAGES:
        for BT in BIAS_TYPES:
            for LLM1 in LLMS:
                for LLM2 in LLMS:

                    if LLM1 == LLM2:
                        continue

                    df1 = df[(df['LLM'] == LLM1)& (df['language'] == lang) & (df['bias_type'] == BT)]
                    df2 = df[(df['LLM'] == LLM2)& (df['language'] == lang) & (df['bias_type'] == BT)]

                    new_df = df1.merge(df2, on=['index'], suffixes=('_1', '_2'))

                    bias1 = list(new_df['bias_1'].values)
                    bias2 = list(new_df['bias_2'].values)

                    print(f'Done for {total_cnt} samples', end='\r')

                    total_cnt += 1
                    try:
                        statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                        ks_results['dimension'].append('LLM')
                        ks_results['control'].append(f'language: {lang}, bias_type: {BT}')
                        ks_results['case1'].append(LLM1)
                        ks_results['case2'].append(LLM2)
                        ks_results['#samples'].append(len(bias1))
                        ks_results['mean_bias_1'].append(np.mean(bias1))
                        ks_results['mean_bias_2'].append(np.mean(bias2))
                        ks_results['statistic'].append(float(statistic))
                        ks_results['pval'].append(float(pval))
                    except Exception as e:
                        error_cnt += 1
                        continue

    for lang in LANGUAGES:
        for LLM in LLMS:
            for BT1 in BIAS_TYPES:
                for BT2 in BIAS_TYPES:
                    if BT1 == BT2:
                        continue

                    df1 = df[(df['LLM'] == LLM)& (df['language'] == lang) & (df['bias_type'] == BT1)]
                    df2 = df[(df['LLM'] == LLM)& (df['language'] == lang) & (df['bias_type'] == BT2)]

                    new_df = df1.merge(df2, on= ['language', 'LLM'], suffixes=('_1', '_2'))
                    
                    bias1 = list(new_df['bias_1'].values)
                    bias2 = list(new_df['bias_2'].values)

                    print(f'Done for {total_cnt} samples', end='\r')

                    total_cnt += 1
                    try:
                        statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                        ks_results['dimension'].append('bias_type')
                        ks_results['control'].append(f'language: {lang}, LLM: {LLM}')
                        ks_results['case1'].append(BT1)
                        ks_results['case2'].append(BT2)
                        ks_results['#samples'].append(len(bias1))
                        ks_results['mean_bias_1'].append(np.mean(bias1))
                        ks_results['mean_bias_2'].append(np.mean(bias2))
                        ks_results['statistic'].append(float(statistic))
                        ks_results['pval'].append(float(pval))
                    except Exception as e:
                        error_cnt += 1
                        continue
    
    for BT in BIAS_TYPES:
        for LLM in LLMS:
            for lang1 in LANGUAGES:
                for lang2 in LANGUAGES:

                    if lang1 == lang2:
                        continue
                        
                    df1 = df[(df['language'] == lang1)& (df['LLM'] == LLM) & (df['bias_type'] == BT)]
                    df2 = df[(df['language'] == lang2)& (df['LLM'] == LLM) & (df['bias_type'] == BT)]

                    new_df = df1.merge(df2, on='index', suffixes=('_1', '_2'))

                    bias1 = list(new_df['bias_1'].values)
                    bias2 = list(new_df['bias_2'].values)

                    total_cnt += 1

                    print(f'Done for {total_cnt} samples', end='\r')
                    try: 
                        statistic, pval = ks_2samp(bias1, bias2, alternative='two-sided')

                        ks_results['dimension'].append('language')
                        ks_results['control'].append(f'LLM: {LLM}, bias_type: {BT}')
                        ks_results['case1'].append(lang1)
                        ks_results['case2'].append(lang2)
                        ks_results['#samples'].append(len(bias1))
                        ks_results['mean_bias_1'].append(np.mean(bias1))
                        ks_results['mean_bias_2'].append(np.mean(bias2))
                        ks_results['statistic'].append(float(statistic))
                        ks_results['pval'].append(float(pval))
                    except Exception as e:
                        error_cnt += 1
                        continue
    

    ks_results_df = pd.DataFrame(ks_results)
    print(f'Error count: {100* error_cnt/total_cnt}')
    ks_results_df.to_csv('ks_results.csv')



def analyse_kstats():

    ks_results = pd.read_csv('ks_results.csv')

    # check the fraction of cases which are significant at 0.05 level depending on the control and dimension

    dimensions = ks_results['dimension'].unique()

    sig_cases  = {'language': ddict(list), 'LLM': ddict(list), 'bias_type': ddict(list)}

    LLM_dict ={
        'meta-llama_Meta-Llama-3-8B': 'L',
        'bigscience_bloom-7b1': 'B',
        'Qwen_Qwen2-7B': 'Q'
    }

    # dimensions = ['language', 'LLM']
    for dim in dimensions:
        for control in ks_results['control'].unique():
            # if 'bias_type' in control: continue
            df = ks_results[(ks_results['dimension'] == dim) & (ks_results['control'] == control)]            

            pvals = df['pval'].values
            
            try:
                reject, pvals_adj, alpha_1, alpha_adj = multipletests(pvals, alpha=0.05, method='bonferroni')

                # reject is an array of boolean values indicating whether the null hypothesis is rejected

                # count the number of cases where reject is True
                print(f'Fraction of significant cases for dimension {dim} and control {control}: {len(df[reject])/len(df)}', end ='\r')

                if control == 'No':
                    continue
                else:
                    other_dims = control.split(',')

                if dim == 'LLM':
                    lang = other_dims[0].replace('language:', '').strip()
                    bt   = other_dims[1].replace('bias_type:', '').strip()

                    # print(f'LLM: lang: {lang}, bias_type: {bt}')
                    sig_cases['LLM']['lang'].append(lang)
                    sig_cases['LLM']['bias_type'].append(bt)
                    sig_cases['LLM']['fraction'].append(len(df[reject])/len(df))
                
                elif dim == 'language':
                    LLM  = other_dims[0].replace('LLM:', '').strip()
                    LLM  = LLM_dict[LLM]
                    bt   = other_dims[1].replace('bias_type:', '').strip()

                    # print(f'language: LLM: {LLM}, bias_type: {bt}')
                    sig_cases['language']['LLM'].append(LLM)
                    sig_cases['language']['bias_type'].append(bt)
                    sig_cases['language']['fraction'].append(len(df[reject])/len(df))

                elif dim == 'bias_type':
                    # LLM  = other_dims[1].replace('LLM:', '').strip().split('_')[1]
                    LLM  = other_dims[1].replace('LLM:', '').strip()
                    LLM  = LLM_dict[LLM]

                    lang = other_dims[0].replace('language:', '').strip()

                    # print(f'bias_type: LLM: {LLM}, lang: {lang}')
                    sig_cases['bias_type']['LLM'].append(LLM)
                    sig_cases['bias_type']['lang'].append(lang)
                    sig_cases['bias_type']['fraction'].append(100*len(df[reject])/len(df))


            except Exception as e:
                continue
    

    sns.set_theme(style='whitegrid', font_scale=1.3, palette='pastel')
    sns.set_context("paper")

    sns.set(rc={"figure.figsize": (9, 2.7)})
    for dimension in sig_cases:
        sig_cases_df = pd.DataFrame(sig_cases[dimension])
        sig_cases_df.to_csv(f'{dimension}_sig_cases.csv')
        if dimension == 'bias_type':
            col = 'lang'
            row = 'LLM'
        elif dimension == 'language':
            col = 'bias_type'
            row = 'LLM'
        elif dimension == 'LLM':
            col = 'lang'
            row = 'bias_type'

        try:
            sig_cases_df = sig_cases_df.pivot(index=row, columns=col, values="fraction")
            ax = sns.heatmap(sig_cases_df, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'label': 'Frac of significant cases'})
            # rotate the elements in the heatmap and the axes

            plt.xticks(rotation=30, fontsize=13)
            plt.yticks(rotation=0, fontsize=13)            
            ax.set_xlabel("")
            ax.set_ylabel("")

            # add a binding box 


            plt.tight_layout()
            plt.savefig(f'{dimension}_inverted_sig_cases_heatmap.pdf')
            plt.close()
            plt.clf()

        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()


    # original code

    # print(f'Plotting the heatmaps for the different dimensions')

    # sns.set_theme(style='whitegrid', font_scale=1.2, palette='pastel')
    # sns.set_context("paper")
    # sns.set(rc={"figure.figsize": (4, 5)})

    # for dimension in sig_cases:
        


    #     sig_cases_df = pd.DataFrame(sig_cases[dimension])
    #     sig_cases_df.to_csv(f'{dimension}_sig_cases.csv')
    #     if dimension == 'bias_type':
    #         row = 'lang'
    #         col = 'LLM'
    #     elif dimension == 'language':
    #         row = 'bias_type'
    #         col = 'LLM'
    #     elif dimension == 'LLM':
    #         row = 'lang'
    #         col = 'bias_type'

    #     try:
    #         sig_cases_df = sig_cases_df.pivot(index=row, columns=col, values="fraction")
    #         sns.heatmap(sig_cases_df, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Fraction of significant cases'})
    #         # add a binding box 

    #         plt.tight_layout()
    #         plt.savefig(f'{dimension}_sig_cases_heatmap.pdf', )
    #         plt.close()
    #         plt.clf()

    #     except Exception as e:
    #         print(e)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--step', type=str, required=True, help='The step to run')

    args = argparser.parse_args()

    if args.step == 'create':
        create_bias_df()
    
    elif args.step == 'anova':
        anova_sigtest()
    
    elif args.step == 'kstest':
        kstest()
    
    elif args.step == 'analyse':
        analyse_kstats()

