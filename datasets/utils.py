from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


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
