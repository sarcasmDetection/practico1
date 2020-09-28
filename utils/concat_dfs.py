import pandas as pd

def get_df(df_paths=[]):
    dfs = [
        pd.read_csv(df_path)
        for df_path in df_paths
    ]

    return pd.concat(dfs)
