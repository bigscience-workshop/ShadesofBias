import os
import sys

import re
import datasets
import pandas as pd
sys.path.append(".")
from config import Config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ALERT_MISSING = False
#NO_TEMPLATES = ['ar', 'zh', 'zh_hant']
LANG_CODES = Config.language_code_list #['_'.join(x.lower().split('-')) for x in Config.language_code_list]
# Where to put the output csv files.
CSV_DIR = '/Users/margaretmitchell/HuggingFace/git/Shades/FormattedBiasShades/by_language/'
ALL_CSV_PATH = '/Users/margaretmitchell/HuggingFace/git/Shades/FormattedBiasShades/all/all.csv'

class CleanDataset:


    @staticmethod
    def convert_to_string(x):
        if x is None:
            return ''
        else:
            return str(x)

    @staticmethod
    def convert_to_list(x):
        #print('x is')
        #print(x)
        if isinstance(x, str):
            new_x = [i.strip() for i in re.split("[,/;]", x)]
            #print("Now it's")
            #print(new_x)
            return new_x
        elif x is None:
            return []
        print("Error -- X was:")
        print(x)
        return x

    @staticmethod
    def get_unique_list_values(counts):
        unique_values = set()
        for index in counts.index:
            for value in index:
                unique_values.add(value)
        return unique_values

    @staticmethod
    def map_to_iso_language_codes(x):
        # Not implemented, since we ended up using the ISO language codes
        # in the raw dataset.
        pass

    @staticmethod
    def map_to_iso_country_codes(x):
        country_iso_map = Config.country_iso_map
        try:
            return [country_iso_map.get(i, i) for i in x]
        except:
            print("Couldn't find country code for country in:")
            print(x)
            return x

    @staticmethod
    def map_to_bool(x):
        try:
            y = x.strip().lower()
            if y[0] == "y" or y == "true":
                return True
            return False
            #elif (
            #    y == "no"
            #    or y == "n"
            #    or y.startswith("x")
            #    or y.startswith("no")
            #    or x == "false"
            #):
            #    return False
            #print("Don't know how to handle this one for the expression:")
            #print(x)
            #return x
        except:
            return False

    def get_col_stats(self, df):
        stats = {}
        for col in df.columns:
            stats[col] = df[col].value_counts()
            logger.info(f"Column: {col}")
            logger.info(stats[col])
            logger.info("Unique Values", len(stats[col]))
            null_pc = (len(df[df[col].isnull()]) / len(df)) * 100
            logger.info("Null %:", null_pc)
            if null_pc != 100 and isinstance(stats[col].index[0], list):
                unique_values = self.get_unique_list_values(stats[col])
                logger.info("Unique Values", unique_values)

    @staticmethod
    def try_strip(x):
        try:
            # Strip if the cell is a string
            if isinstance(x, str):
                return x.strip()
        except Exception as e:
            # Just return the original value if there's an error
            logger.warning(f"Error processing: {x}, Error: {e}")
        return x

def combine_lists(row):
    original_languages = row["stereotype_origin_langs"]
    valid_languages = row["stereotype_valid_langs"]
    print(original_languages)
    print(valid_languages)
    return list(set(original_languages + valid_languages))

def copy_original_language_to_valid_language(df):
    df['stereotype_valid_langs'] = df.apply(combine_lists, axis=1)
    return df

def copy_original_content_to_contrasts(df, index):
    """Copies content from the _original statement to the contrasts:
    bias_type and templates. Moves on when something is missing."""
    df_mini = df[df['index'] == index]
    subset_ids = list(df_mini.loc[df_mini['subset'] != "_original"]['subset'])
    for letter_idx in range(len(subset_ids)):
        letter = subset_ids[letter_idx]
        # If the bias_type isn't there, fill it in.
        original_bias_type = df_mini[df_mini['subset'] == '_original']["bias_type"].values
        try:
            df.loc[(df['index'] == index) & (df['subset'] == letter), "bias_type"] = original_bias_type
        except ValueError:
            logger.warning("Issue with indexing for %s, %s" % (index, letter))
    # Copy the template from the last-completed one to the blank cells below it.
    for lang in LANG_CODES:
            lang = lang.lower()
            #if lang not in NO_TEMPLATES:
            try:
                prev_template = df_mini.loc[df_mini['subset'] == '_original'][lang + "_templates"]
            except KeyError:
                continue
            if not prev_template.any() and ALERT_MISSING:
                logger.warning("ISSUE: No original template for language %s, index %s," % (lang, index))
                continue
            for letter_idx in range(len(subset_ids)):
                letter = subset_ids[letter_idx]
                next_template = df_mini[df_mini['subset'] == letter][lang + "_templates"]
                if not next_template.any():
                    try:
                        df.loc[(df['index'] == index) & (df['subset'] == letter), lang + "_templates"] = prev_template.values
                    except ValueError:
                        logger.warning("Issue with indexing for %.1f, %s" % (index, letter))
                else:
                    prev_template = next_template
    return df

def convert_dataset(col_map_path, df):
    """
    Normalizes the content across rows, separates dataset into different languages.
    :param col_map_path: csv file containing data about each column and how it should be normalized.
    :param df: original dataset
    :return: organized, normalized, cleaned dataset
    """
    cleaner = CleanDataset()
    df = df.applymap(cleaner.try_strip)

    # Create dict from the column specifications.
    column_map = pd.read_csv(col_map_path)
    column_map = column_map[~column_map["new_column_name"].isnull()]
    column_map_dict = dict(
        zip(column_map["current_column_name"], column_map["new_column_name"])
    )
    # Rename the columns following the column specifications.
    df = df.rename(columns=column_map_dict)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.reset_index(drop=True, inplace=True)

    # Format bias_type column as lists
    df["bias_type"] = df["bias_type"].apply(cleaner.convert_to_list)
    logger.info(df.columns)

    # Fill out all bias_type and template cells possible.
    indices = df['index'].unique()
    for index in indices:
        df = copy_original_content_to_contrasts(df, index)

    # Do more basic formatting.
    df["stereotype_origin_langs"] = df["stereotype_origin_langs"].apply(
        cleaner.convert_to_list
    )
    df["stereotype_valid_langs"] = df["stereotype_valid_langs"].apply(
        cleaner.convert_to_list
    )
    df = copy_original_language_to_valid_language(df)

    df["stereotype_valid_regions"] = df["stereotype_valid_regions"].apply(
        cleaner.convert_to_list
    )
    df["stereotype_valid_regions"] = df["stereotype_valid_regions"].apply(
        cleaner.map_to_iso_country_codes
    )


    for col in df.columns:
        if "expression" in col:
            df[col] = df[col].apply(cleaner.map_to_bool)
        if "comments" in col:
            df[col] = df[col].apply(cleaner.convert_to_string)
    df.reset_index(drop=True, inplace=True)
    #cleaner.get_col_stats(
    #    df
    #)  # Print Null stats, #todo unique categorical values (some need to be manually mapped)
    return df

def main(
    raw_dataset_path="LanguageShades/BiasShadesRaw",
    col_map_fid="BiasShades_fields - columns.csv",
):
    # col_map_path assumed to be in the same directory as the current file.
    col_map_path = os.path.dirname(os.path.realpath(__file__)) + "/" + col_map_fid
    # file is https://docs.google.com/spreadsheets/d/1dyEYmsGW3i1MpSoKyuPofpbjqEkifUhdd19vVDQr848/edit?usp=sharing
    df = datasets.load_dataset(raw_dataset_path, split="train").to_pandas()
    df = convert_dataset(col_map_path, df=df)
    df.to_csv(ALL_CSV_PATH, index=False)
    # TODO: Changing to dataset_dict doesn't even need to happen anymore.
    #dataset_dict = datasets.Dataset.from_pandas(df)
    basic_cols = Config.basic_cols
    #dataset_dict_test = {} #dataset_dict.select_columns(basic_cols)}
    for lang in LANG_CODES:
        print(lang)
        #if lang not in NO_TEMPLATES:
        #    lang_cols += [lang + "_templates"]
        lang_cols = basic_cols + [lang + "_biased_sentences", lang + "_expression", lang + "_comments"]
        #print(df)
        try:
            df[lang_cols].to_csv(CSV_DIR + lang + '.csv', index=False)
            print("Saved to %s" % CSV_DIR)
        except KeyError:
            print("No columns for %s" % lang)
        #pd.DataFrame(dataset_dict.select_columns(lang_cols)).to_csv(CSV_DIR + lang + '.csv', index=False)
    #dataset_dict_test_hub = datasets.Dataset.from_dict(dataset_dict_test)
    #dataset_dict_test_hub.push_to_hub(formatted_dataset_upload_path)


if __name__ == "__main__":
    main()
