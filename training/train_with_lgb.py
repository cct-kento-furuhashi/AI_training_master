import os
import pickle
import warnings

import lightgbm as lgb
import pandas as pd

from datasets.utils import read_df
from settings.params import LGB_MODEL_PATH, N_BOOST_ROUND, N_EARLY_STOPPING_ROUND, TRAIN_PARAMS

warnings.simplefilter("ignore")


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, model_idx: int) -> lgb.Booster:
    """train LGBMで訓練を行う

    Args:
        train_df (pd.DataFrame): 訓練データ
        valid_df (pd.DataFrame): テストデータ
        model_idx (int): モデルID

    Returns:
        lgb.Booster: モデル
    """
    model_path = LGB_MODEL_PATH.format(model_idx)
    if os.path.exists(model_path):
        with open(model_path, "rb") as rf:
            model = pickle.load(rf)
        return model
    # XとYに分ける
    train_x, train_y = read_df(train_df, "medv")
    valid_x, valid_y = read_df(valid_df, "medv")

    # データセット作成
    train_dataset = lgb.Dataset(train_x, train_y)
    valid_dataset = lgb.Dataset(valid_x, valid_y)

    # 訓練
    model = lgb.train(
        TRAIN_PARAMS,
        train_dataset,
        valid_sets=valid_dataset,
        num_boost_round=N_BOOST_ROUND,
        early_stopping_rounds=N_EARLY_STOPPING_ROUND,
        verbose_eval=False,
    )
    with open(model_path, "wb") as wf:
        pickle.dump(model, wf)
    return model
