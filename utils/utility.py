import os
import random
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import pandas as pd


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """時間測定

    Args:
        name (str): 表示名

    Yields:
        Generator[None, None, None]: returnなし

    Example:
        with timer("sample"):
            time.sleep(2.0)
        # [sample] done in 2 s
    """
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.1f} s")


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのメモリ量を下げるために型を変更する

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: 型変更した後のDataFrame
    """
    # start_mem = df.memory_usage().sum() / 1024**2
    # print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            # int型なら
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            # float型なら
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    # end_mem = df.memory_usage().sum() / 1024**2
    # print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    # print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def set_seed(seed: int = 815) -> None:
    """再現性確保のためのシード固定

    Args:
        seed (int, optional): シード値
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
