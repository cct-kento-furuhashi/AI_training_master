from typing import Dict, Union

import optuna.integration.lightgbm as lgb
import pandas as pd

from datasets.utils import read_df
from settings.params import N_BOOST_ROUND, N_EARLY_STOPPING_ROUND, SEED, TARGET_OBJ


def train_with_optimize_param(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> lgb.Booster:
    """パラメータ最適化+訓練を行う

    Args:
        train_df (pd.DataFrame): 訓練データ
        valid_df (pd.DataFrame): 検証用データ

    Returns:
        lgb.Booster: モデル
    """
    train_x, train_y = read_df(train_df, TARGET_OBJ)
    valid_x, valid_y = read_df(valid_df, TARGET_OBJ)

    train_dataset = lgb.Dataset(train_x, train_y)
    valid_dataset = lgb.Dataset(valid_x, valid_y)

    params: Dict[str, Union[int, float, str]] = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose_eval": False,
    }

    model = lgb.train(
        params,
        train_dataset,
        valid_sets=valid_dataset,
        num_boost_round=N_BOOST_ROUND,
        early_stopping_rounds=N_EARLY_STOPPING_ROUND,
    )
    return model
