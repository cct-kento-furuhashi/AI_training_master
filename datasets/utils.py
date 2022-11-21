import pandas as pd
import numpy as np
from typing import Tuple
import numpy.typing as npt

def read_df(df:pd.DataFrame, target_column: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    xs = df.drop(target_column, axis=1).values
    ys = df.loc[:,target_column].values
    return xs,ys