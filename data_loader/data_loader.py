from typing import Any
import pandas as pd


def read_csv(csv_path:str)->pd.DataFrame:
    """read_csv
    CSVを読み込み、CSVの中身を～～～型で返す。

    Args:
        csv_path (str): csvのおいてあるパス

    Returns:
        pd.DataFrame: データフレーム型
    """
    csv_data = pd.read_csv(csv_path)
    
    return csv_data.drop(csv_data.columns[[0]],axis=1)
