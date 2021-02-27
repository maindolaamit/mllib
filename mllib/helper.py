from glob import glob

import numpy as np
import pandas as pd


def get_dir_dataframe(path, pattern="*.csv"):
    """
    This methods reads all files in the given directory. Reads the file and return a Pandas DataFrame
    concatenating all the files into a single df
    :return: Pandas DataFrame
    """
    files = sorted(glob(f"{path}/{pattern}"))
    df = pd.concat((pd.read_csv(file) for file in files), index=False)

    return df


def get_stugres_bin(sample_size):
    """Return the number of bins for sample size based on Sturge's rule"""
    return int(np.floor(np.log2(sample_size) + 1))


def get_zero_var_cols(df):
    """
    Method to get all columns having zero variance
    """
    variance = df.var()
    # Check for numerical columns
    zero_var_columns = [variance[variance == 0].index.to_list()]
    # Check for Feature columns
    columns = df.select_dtypes(include='object').columns.to_list()
    for column in columns:
        if df[column].nunique == 1:
            zero_var_columns.append(column)

    return zero_var_columns


def get_columns_by_variance(X, y, threshold=0.0):
    """
    Method to fetch columns based on threshold variance
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold).fit(X, y)
    return X.columns[selector.get_support()]


def drop_columns_by_variance(df, threshold=0):
    """
    Method to drop all columns having zero variance
    """
    columns = get_columns_by_variance(df, threshold)
    drop_columns = set(df.columns) - set(columns)
    print(f"Dropping columns : {drop_columns}")
    return df.drop(drop_columns, axis=1)


def get_display_time(elapsed_seconds):
    """
    Returns the readable format of Time elapsed in Hours, Min and seconds
    :param elapsed_seconds:
    :return:
    """
    hours, minutes, sec = 0, 0, 0
    import math
    if elapsed_seconds > 60 * 60:
        hours = math.floor(int(elapsed_seconds / 3600))
        minutes = math.floor(int(elapsed_seconds - hours * 3600) / 60)
        sec = math.round(elapsed_seconds - hours * 3600 - minutes * 60)
        return f"{hours} Hour {minutes} Minutes {sec} seconds"
    elif elapsed_seconds > 60:
        minutes = math.floor(int(elapsed_seconds / 60))
        sec = round(elapsed_seconds - minutes * 60)
        return f"{minutes} Minutes {sec} seconds"
    else:
        sec = round(elapsed_seconds)
        return f"{sec} seconds"


# def get_uniq_label_counts(df):
#     """
#     Get all unique labels count in the DataFrame
#     """
#     object_columns = get_dtype_columns(df)
#     return list(map(lambda x: (x, len(df[x].unique())), object_columns))
#
#
# def get_col_top_labels(series, top=10):
#     """
#     Get top n labels for the given Pandas Series
#     """
#     index = series.value_counts().head(top).index
#     return index.to_list()
#
#
# def top_one_hot_columns(df, top=10):
#     """
#     Returns a DataFrame of One Hot Encoded values of all Object columns
#     """
#     object_columns = get_dtype_columns(df)
#     one_hot_df = pd.DataFrame()
#     for col in object_columns:
#         for label in get_col_top_labels(df[col], top):
#             one_hot_col = str(col + '_' + label)
#             series = df[col]
#             if one_hot_col not in one_hot_df.columns.to_list():
#                 one_hot_df[one_hot_col] = np.where(series == label, 1, 0)
#
#     return one_hot_df

if __name__ == '__main__':
    None
