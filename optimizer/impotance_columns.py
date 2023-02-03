from typing import List

import numpy as np
from pandas import DataFrame

from datasets.utils import iter_kfold
from settings.params import N_SPLITS, TARGET_OBJ
from training.train_with_lgb import train


def get_top_importance_columns(data_df: DataFrame, n: int = 5) -> List[str]:
    """モデルを訓練し、上位N個のカラムを取得する

    Args:
        data_df (DataFrame): 使用するデータ
        n (int): 上位n個のカラムを取得する

    Returns:
        List[str]: カラム
    """
    column_num = data_df.shape[1] - 1  # 目的変数を除く
    if n >= column_num:
        n = column_num
    # 結果保持
    result_columns = [c for c in data_df.columns.values if c != TARGET_OBJ]
    result_columns.insert(0, "idx")
    result_list: List[List[int]] = list()

    # シャッフル分割交差検証を行う
    for idx, (train_data, valid_data) in enumerate(iter_kfold(data_df, n_splits=N_SPLITS)):
        # 訓練
        model = train(train_data, valid_data, is_save=False)

        # feature importance算出
        importance = model.feature_importance()
        result = [idx]
        result.extend(importance)
        result_list.append(result)

    # 結果作成
    result_df = DataFrame(result_list, columns=result_columns, dtype=np.uint16)
    result_df = result_df.set_index("idx")

    sum_result_df = result_df.sum()

    # 上位5つのカラムを取得
    importance_columns = sorted(sum_result_df.items(), key=lambda x: x[1], reverse=True)  # type: ignore
    importance_list = [c for c in np.array(importance_columns)[:n, 0]]
    # 目的変数を追加
    importance_list.append(TARGET_OBJ)
    return importance_list
