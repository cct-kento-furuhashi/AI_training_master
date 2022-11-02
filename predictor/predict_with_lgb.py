from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datasets.utils import read_df
from settings.params import TARGET_OBJ


def predict(test_df: pd.DataFrame, model: lgb.Booster) -> Tuple[float, float, float]:
    """predict 予測

    Args:
        test_df (pd.DataFrame): テストデータ
        model (lgb.Booster): モデル

    Returns:
        Tuple[float, float, float]: RMSE(二乗平均平方根誤差), MAE(平均絶対誤差), R2スコア
    """
    test_x, test_y = read_df(test_df, TARGET_OBJ)

    pred_y = model.predict(test_x)

    r2 = r2_score(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))

    return rmse, mae, r2
