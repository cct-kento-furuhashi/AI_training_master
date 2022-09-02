import os
from logging import warning

import pandas as pd


def _read_csv(csv_path: str) -> pd.DataFrame:
    """_read_csv CSVファイルを読み込んでデータを返す

    Args:
        csv_path (str): CSVのパス

    Returns:
        pd.DataFrame: データ
    """
    df = pd.read_csv(csv_path)
    return df

def read_data(file_path: str) -> pd.DataFrame:
    """read_data ファイルを読み込んでデータを返す

    Args:
        file_path (str): 入力データのパス

    Returns:
        pd.DataFrame: データ
    """
    _, ext = os.path.splitext(file_path)
    if ext == ".csv":
        df = _read_csv(file_path)
    else:
        warning("読み取れないファイルです")
        raise Exception
    return df
