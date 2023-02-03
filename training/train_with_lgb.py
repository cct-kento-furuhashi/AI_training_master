import os
import pickle
import warnings
from typing import Dict, Union

import lightgbm as lgb
import pandas as pd

from datasets.utils import read_df
from settings.params import LGB_MODEL_PATH, N_BOOST_ROUND, N_EARLY_STOPPING_ROUND, TARGET_OBJ, TRAIN_PARAMS

warnings.simplefilter("ignore")


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: Dict[str, Union[int, float, str]] = TRAIN_PARAMS,
    model_idx: int = 0,
    is_save: bool = True,
) -> lgb.Booster:
    """train LGBMで訓練を行う

    Args:
        train_df (pd.DataFrame): 訓練データ
        valid_df (pd.DataFrame): テストデータ
        params (Dict[str, Union[int, float, str]]): 訓練パラメータ
        model_idx (int): モデルID
        is_save (bool): モデルを保存するか

    Returns:
        lgb.Booster: モデル
    """
    model_path = LGB_MODEL_PATH.format(model_idx)
    if os.path.exists(model_path):
        with open(model_path, "rb") as rf:
            model = pickle.load(rf)
        return model
    # XとYに分ける
    train_x, train_y = read_df(train_df, TARGET_OBJ)
    valid_x, valid_y = read_df(valid_df, TARGET_OBJ)

    # データセット作成
    train_dataset = lgb.Dataset(train_x, train_y)
    valid_dataset = lgb.Dataset(valid_x, valid_y)

    # 訓練
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=valid_dataset,
        num_boost_round=N_BOOST_ROUND,
        early_stopping_rounds=N_EARLY_STOPPING_ROUND,
        verbose_eval=False,
    )
    if is_save:
        with open(model_path, "wb") as wf:
            pickle.dump(model, wf)
    return model
