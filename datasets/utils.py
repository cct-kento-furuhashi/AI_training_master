import pandas as pd
import numpy as np
from typing import Tuple
import numpy.typing as npt
from itertools import combinations

def read_df(df:pd.DataFrame, target_column: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    xs = df.drop(target_column, axis=1).values
    ys = df.loc[:,target_column].values
    return xs,ys

def get_columns(columns:list,target_column: str) -> list:
    columns.remove(target_column)
    for v in combinations(columns,5):
        selected_columns = list(v)
        selected_columns.append(target_column)
        yield selected_columns






