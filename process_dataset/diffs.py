import pandas as pd

def load_data(file_path):
    """Load data from file based on extension."""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .csv, or .tsv.")

def normalize_value(value):
    """Normalize values for comparison."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value

def compare_translations(file1, file2):
    """Compare two files row by row and column by column."""
    df1 = load_data(file1)
    df2 = load_data(file2)

    # Standardize column names
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    if len(df1) != len(df2):
        print("The files have a different number of rows.")
        return

    for idx in df1.index:
        row1 = df1.loc[idx]
        row2 = df2.loc[idx]

        # Context from 'index' and 'subset' columns
        index_val = row1.get('index', 'N/A')
        subset_val = row1.get('subset', 'N/A')

        for col in df1.columns:
            if col not in df2.columns:
                print(f"Column missing - Row {idx}, Column '{col}'")
                continue

            cell1 = normalize_value(row1[col])
            cell2 = normalize_value(row2[col])

            if cell1 is None and cell2 is None:
                continue

            if cell1 is not None and cell2 is None:
                print(f"Possible deletion - Row {idx}, Index {index_val}, Subset {subset_val}, Column '{col}': was '{cell1}', now empty.")

            elif cell1 is None and cell2 is not None:
                print(f"Unexpected input - Row {idx}, Index {index_val}, Subset {subset_val}, Column '{col}': was empty, now '{cell2}'.")

            elif 'templates' in col and cell1 != cell2:
                print(f"Change in template - Row {idx}, Index {index_val}, Subset {subset_val}, Column '{col}': '{cell1}' -> '{cell2}'")

            elif 'biased sentences' in col:
                eng_column = 'english: biased sentences'
                if eng_column in df1.columns and normalize_value(df1.loc[idx, eng_column]) != normalize_value(df2.loc[idx, eng_column]) and cell1 == cell2:
                    print(f"Misalignment in translation - Row {idx}, Index {index_val}, Subset {subset_val}, Column '{col}': English text changed but not in {col}")


#Data
file1 = '/Users/timmdill/Documents/GitHub/QT/Data/Shades (1).xlsx'
file2 = '/Users/timmdill/Documents/GitHub/QT/Data/Shades (2).xlsx'

compare_translations(file1, file2)
