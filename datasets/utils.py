from itertools import combinations
from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import KFold

from settings.params import TARGET_OBJ


def read_df(df: pd.DataFrame, target_col: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """read_df データをXとYに分ける

    Args:
        df (pd.DataFrame): 分けるデータ
        target_col (str): 目的変数の列名

    Returns:
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: XとY
    """
    xs = df.drop(target_col, axis=1).values
    ys = df.loc[:, target_col].values
    return xs, ys


def iter_columns(df: pd.DataFrame, k: int = 5) -> Iterator[List[str]]:
    """使用するカラムの全組み合わせを取得する

    Args:
        df (pd.DataFrame): 使用データ
        k (int, optional): 何個の列を使用するか。Defaults to 5

    Yields:
        Iterator[List[str]]: 使用する列のリストを返す
    """
    # 対象を除いた列のリストを作成
    columns = [c for c in df.columns.values if c != TARGET_OBJ]
    # 全組み合わせ
    for use_columns in combinations(columns, 5):
        use_columns_list: List[str] = list(use_columns)
        # 対象を追加
        use_columns_list.append(TARGET_OBJ)
        yield use_columns_list


def iter_kfold(
    df: pd.DataFrame, n_splits: int = 5, is_shuffle: bool = True, random_state: int = 0
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """交差検証のデータを返す

    Args:
        df (pd.DataFrame): 使用データ
        n_splits (int, optional): 分割数。Defaults to 5
        is_shuffle (bool, optional): データをシャッフルするかどうか。Defaults to True
        random_state (int, optional): ランダムシード。Defaults to 0

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 訓練データとテストデータ
    """
    kf = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=random_state)
    for train_index, test_index in kf.split(df):
        yield df.iloc[train_index], df.iloc[test_index]
