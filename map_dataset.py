import re
import sys
import datasets
import pandas as pd
sys.path.append(".")
from config import Config

NO_TEMPLATES = ['ar', 'zh', 'zh_hant']

class CleanDataset:

    @staticmethod
    def convert_to_list(x):
        if isinstance(x, str):
            return [i.strip() for i in re.split("[,/]", x)]
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
        pass

    @staticmethod
    def map_to_iso_country_codes(x):
        country_iso_map = {
            "Mainland China": "CN",
            "India?": "IN",
            "Brazill": "BR",
            "Uzbekistan": "UZ",
            "Dominican Republic": "DO",
            "Romania": "RO",
            "Russia": "RU",
            "Hong Kong": "HK",
            "France": "FR",
            "Netherlands": "NL",
            "Flemish Belgium": "BE",  # Assuming it's referring to Belgium
            "Poland": "PL",
            "Italy": "IT",
            "India": "IN",
            "France?": "FR",
            "Japan": "JP",
            "Brazil": "BR",
            "West Germany": "DE",  # West Germany is now part of Germany (DE)
            "Flanders Belgium": "BE",  # Assuming it's referring to Belgium
            "China": "CN",
            "Germany": "DE",
            "mainland China": "CN",
            "Lebanon": "LB",
            "US": "US",
        }
        try:
            return [country_iso_map.get(i, i) for i in x]
        except:
            return x

    @staticmethod
    def map_to_bool(x):
        try:
            y = x.strip().lower()
            if y == "yes" or y == "y" or y == "yes." or y == "true":
                return "True"
            elif (
                y == "no"
                or y == "n"
                or y.startswith("x")
                or y.startswith("no")
                or x == "false"
            ):
                return "False"
            return x
        except:
            return x

    def get_col_stats(self, df):
        stats = {}
        for col in df.columns:

            stats[col] = df[col].value_counts()
            print(f"Column: {col}")
            print(stats[col])
            print("Unique Values", len(stats[col]))
            null_pc = (len(df[df[col].isnull()]) / len(df)) * 100
            print("Null %:", null_pc)
            if null_pc != 100 and isinstance(stats[col].index[0], list):
                unique_values = self.get_unique_list_values(stats[col])
                print("Unique Values", unique_values)

    @staticmethod
    def try_strip(x):
        try:
            # Strip if the cell is a string
            if isinstance(x, str):
                return x.strip()
        except Exception as e:
            # Just return the original value if there's an error
            print(f"Error processing: {x}, Error: {e}")
        return x


# Apply the function to each element in the DataFrame


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

    # Normalize how the bias_type column values appear
    df["bias_type"] = df["bias_type"].apply(cleaner.convert_to_list)

    # Copy the bias_type and template values from the _original to the contrasts.
    # If there isn't a contrast, print the issue & continue.
    indices = df['index'].unique()
    lang_codes = Config.language_codes.values()
    # Most statements have a single contrast marked 'a'. Some have more.
    # This allows for a lot more, if they are there.
    contrast_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    for index in indices:
        #try:
        #    # Copy the bias_type across contrasts
        #    df.loc[(df['index'] == index) & (df['subset'] == 'a'), "bias_type"] = df.loc[(df['index'] == index) &
        #(df['subset'] == '_original'), "bias_type"].values
        # #   # Copy the template across contrasts, for each language.
        # #   # If there isn't a template for an _original, continue.
        #  #  for lang in lang_codes:
        #   #     try:
        #            df.loc[(df['index'] == index) & (df['subset'] == 'a'), lang + "_templates"] = df.loc[(df[
        #            'index'] == index) & (df['subset'] == '_original'), lang + "_templates"].values
        #        except KeyError:
        #            continue
        #except ValueError:
        #    print("Missing pairing for %.1f" % index)
        #    continue
        df = copy_original_content_to_contrasts(df, index, lang_codes, contrast_letters)
    print(df)
    #sys.exit()
    #df.loc[df['index'] == 1.0][df['Subset'] == 'a']["bias_type"] = "poop" #df[df['index'] == 1.0][df['Subset'] == '_original']["bias_type"]
    #print(df[df['index'] == 1.0][df['Subset'] == 'a']["bias_type"])
    #sys.exit()
    #print(df.loc[df['index'] == 1.0][df['Subset'] == 'a']["bias_type"])
    #print(df[df['index'] == 1.0])
    #sys.exit()
    #df.loc[df['Subset'] == 'a', "bias_type"] = df.loc[df['Subset'] == '_original', "bias_type"]
    #print(df[df['Subset'] == '_original']['index'])
    #indices = df['index'].unique()
    #print(indices)
    #print(df[df["Subset"] == "a"]["bias_type"])
    #df.loc[:, df["Subset"] == "a"]["bias_type"] = df[df["Subset"] == "_original"]["bias_type"]
    #print(df[df["Subset"] == "a"]["bias_type"])
    #for idx in indices:
    #    print(df.loc[df['index' == idx], "Subset"])
    #print(df)
    #sys.exit()

    #sys.exit()

    df["stereotype_origin_langs"] = df["stereotype_origin_langs"].apply(
        cleaner.convert_to_list
    )
    df["stereotype_valid_langs"] = df["stereotype_valid_langs"].apply(
        cleaner.convert_to_list
    )
    df["stereotype_valid_regions"] = df["stereotype_valid_regions"].apply(
        cleaner.convert_to_list
    )

    # df['stereotype_origin_langs'] = df['stereotype_origin_langs'].apply(map_to_iso_language_codes)
    # df['stereotype_valid_langs'] = df['stereotype_valid_langs'].apply(map_to_iso_language_codes)
    df["stereotype_valid_regions"] = df["stereotype_valid_regions"].apply(
        cleaner.map_to_iso_country_codes
    )
    for col in df.columns:
        if "perceived" in col or "expression" in col:
            df[col] = df[col].apply(cleaner.map_to_bool)
    df.reset_index(drop=True, inplace=True)
    cleaner.get_col_stats(
        df
    )  # Print Null stats, #todo unique categorical values (some need to be manually mapped)
    return df


def copy_original_content_to_contrasts(df, index, lang_codes, contrast_letters):
    """Copies content from the _original statement to the contrasts:
    bias_type and templates. Moves on when something is missing."""
    # TODO: Check if the letter is there first, only run if so.
    for letter in contrast_letters:
        try:
            original_bias_type = df.loc[(df['index'] == index) & (df['subset'] == '_original'), "bias_type"].values
            df.loc[(df['index'] == index) & (df['subset'] == letter), "bias_type"] = original_bias_type
        except ValueError:
            if letter == 'a':
                print("Missing contrast pairing for %.1f" % index)
            pass # This can probably just return, as the next statement would also fail.
        # Copy the template from the _original to the contrasts, for each language.
        # If the _original template isn't there, move on.
        for lang in lang_codes:
            if lang not in NO_TEMPLATES:
                try:
                    original_template = df.loc[(df['index'] == index) & (df['subset'] == '_original'), lang + "_templates"].values
                    # TODO: Do this ONLY if a template is not already there.
                    # If it is, assert that it is what is expected from the _original_template.
                    df.loc[(df['index'] == index) & (df['subset'] == letter), lang + "_templates"] = original_template
                except Exception as e:
                    print("Missing template for language %s, index %.1f" % (lang, index))
                    print("Error:")
                    print(e)
                    pass
    return df


if __name__ == "__main__":
    # file is https://docs.google.com/spreadsheets/d/1dyEYmsGW3i1MpSoKyuPofpbjqEkifUhdd19vVDQr848/edit?usp=sharing
    df = datasets.load_dataset("LanguageShades/BiasShadesRaw", split="train").to_pandas()
    # TODO: Check how this columns.csv should be updated.
    converted_df = convert_dataset("BiasShades_fields - columns.csv", df=df)
    hub_dataset_tmp = datasets.Dataset.from_pandas(converted_df)
    # TODO: Break this down by language.
    hub_dataset = datasets.DatasetDict({"test": hub_dataset_tmp})
    from huggingface_hub import HfApi, HfFolder

    # Replace 'your-username' with your Hugging Face username
    dataset_name = "LanguageShades/FormattedBiasShades"

    # Push the dataset to the hub
    hub_dataset.push_to_hub(dataset_name)
