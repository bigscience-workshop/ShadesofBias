#### Format the dataset into a tsv
import os
import pdb
import glob
import pandas as pd

UTF8_LANG = {"zh"}

def format_dataset(instances, lang='en'):
    """
    instances: a list of instances to be formated, presumably all of them are stereotype statements
    return: a pandas dataframe with each column be the formatted data
    """
    res = []
    prompt_files = glob.glob(os.path.join(f"prompts_{lang}", "*.txt"))
    prompts = dict()
    for prompt_file in prompt_files:
        # Load a list of available prompts
        curr_prompt = open(prompt_file, "r")
        prompts[prompt_file.split('/')[-1].split('.txt')[0]] = curr_prompt.read()
        curr_prompt.close()

    for idx, instance in enumerate(instances["text"]):
        # Write each prompt into the dict
        curr_intance_dict = {"idx": idx, "original": instance}
        for prompt_idx in prompts:
            print(prompts[prompt_idx])
            curr_intance_dict[prompt_idx] = prompts[prompt_idx].format(input=instance)
            if lang in UTF8_LANG:
                curr_intance_dict[prompt_idx] = prompts[prompt_idx].format(input=instance)

        res.append(curr_intance_dict)

    res = pd.DataFrame(res)
    print(res.to_json(f"formatted_dataset_{lang}.jsonl", orient='records', lines=True, force_ascii=False))
    return res



if __name__ == "__main__":
    lang = 'zh'
    instances = pd.read_csv(f"raw_data_{lang}.csv")
    format_dataset(instances, lang=lang)
