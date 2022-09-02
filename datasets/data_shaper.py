from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def shape_datas(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """shape_datas データ整形

    Args:
        df (pd.DataFrame): CSVデータ

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 整形後のデータ(訓練データ, 検証データ, テストデータ)
    """
    train_valid_df, test_df = train_test_split(df, test_size=56, shuffle=False)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, shuffle=False)
    return train_df, valid_df, test_df
