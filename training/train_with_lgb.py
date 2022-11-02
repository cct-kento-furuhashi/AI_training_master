import os
import pickle

import lightgbm as lgb
import pandas as pd

from datasets.utils import read_df
from settings.params import LGB_MODEL_PATH, TRAIN_PARAMS


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> lgb.Booster:
    """train LGBMで訓練を行う

    Args:
        train_df (pd.DataFrame): 訓練データ
        valid_df (pd.DataFrame): テストデータ

    Returns:
        lgb.Booster: モデル
    """
    if os.path.exists(LGB_MODEL_PATH):
        with open(LGB_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    # XとYに分ける
    train_x, train_y = read_df(train_df, "medv")
    valid_x, valid_y = read_df(valid_df, "medv")

    # データセット作成
    train_dataset = lgb.Dataset(train_x, train_y)
    valid_dataset = lgb.Dataset(valid_x, valid_y)

    # 訓練
    model = lgb.train(
        TRAIN_PARAMS, train_dataset, valid_sets=valid_dataset, num_boost_round=1000, early_stopping_rounds=100
    )
    with open(LGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model
