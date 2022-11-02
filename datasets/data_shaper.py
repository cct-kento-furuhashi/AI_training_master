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


def split_test_data(
    df: pd.DataFrame, train_size: float = 0.8, test_size: float = 0.2, is_shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split_test_data 訓練に使用するデータとテストに使用するデータを分ける

    Args:
        df (pd.DataFrame): 使用するデータ
        train_size (float, optional): 訓練データの割合。Defaults to 0.8
        test_size (float, optional): テストデータの割合。Defaults to 0.2
        is_shuffle (bool, optional): データをシャッフルするかどうか。Defaults to True

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 訓練データ, テストデータ
    """
    train_df, test_df = train_test_split(df, train_size=train_size, test_size=test_size, shuffle=is_shuffle)
    return train_df, test_df
