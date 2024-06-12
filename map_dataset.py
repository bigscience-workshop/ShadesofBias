import re

import datasets
import pandas as pd


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


def convert_dataset(col_map_path):
    cleaner = CleanDataset()
    df = datasets.load_dataset("LanguageShades/BiasShades", split="train").to_pandas()

    column_map = pd.read_csv(col_map_path)

    column_map = column_map[~column_map["new_column_name"].isnull()]
    column_map_dict = dict(
        zip(column_map["current_column_name"], column_map["new_column_name"])
    )

    df = df.rename(columns=column_map_dict)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.reset_index(drop=True, inplace=True)

    df["bias_type"] = df["bias_type"].apply(cleaner.convert_to_list)

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


if __name__ == "__main__":
    # file is https://docs.google.com/spreadsheets/d/1dyEYmsGW3i1MpSoKyuPofpbjqEkifUhdd19vVDQr848/edit?usp=sharing
    df = convert_dataset("BiasShades_fields - columns.csv")
    # df = datasets.Dataset.from_pandas(df)
    # df = datasets.DatasetDict({"test": df})
    # from huggingface_hub import HfApi, HfFolder

    # # Replace 'your-username' with your Hugging Face username
    # dataset_name = "jordiclive/test-shades"

    # # Push the dataset to the hub
    # df.push_to_hub(dataset_name)
