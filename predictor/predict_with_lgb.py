import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from datasets.utils import read_df


def predict(test_df: pd.DataFrame, model: lgb.Booster) -> None:
    """predict 予測

    Args:
        test_df (pd.DataFrame): テストデータ
        model (lgb.Booster): モデル
    """
    test_x, test_y = read_df(test_df, "medv")

    pred_y = model.predict(test_x)

    r2 = r2_score(test_y, pred_y)
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))

    print(f"Score --- R2: {r2:.2f} RMSE: {rmse:.2f}")
