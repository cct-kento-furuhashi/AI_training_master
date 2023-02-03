import os
from copy import deepcopy
from typing import List, Union

from pandas import DataFrame

from data_loader.data_loader import read_data
from datasets.data_shaper import split_test_data
from datasets.utils import iter_kfold
from optimizer.impotance_columns import get_top_importance_columns
from predictor.predict_with_lgb import predict
from settings.params import INPUT_PATH, N_SPLITS, TARGET_OBJ, USE_COLUMN_NUM
from training.train_with_lgb import train
from utils.utility import set_seed

"""
5個の列を使い、一番良い組み合わせを求める。
lightGBMの機能を使う
"""

if __name__ == "__main__":
    set_seed()
    # ファイル読み込み
    load_data = read_data(INPUT_PATH)

    # DEBUG: データ確認
    print(f"データ数{load_data.shape}\n列: {load_data.columns.values}")

    # Testデータ取得
    load_data, test_data = split_test_data(load_data)

    # 使用するカラムを取得
    use_columns = get_top_importance_columns(load_data, n=USE_COLUMN_NUM)
    load_data = load_data.loc[:, use_columns]
    test_data = test_data.loc[:, use_columns]

    # 結果格納用
    result_list: List[List[Union[float, int, str]]] = list()
    result_columns: List[str] = [f"col{i+1}" for i in range(len(use_columns) - 1)]
    result_columns.extend(["kfold_idx", "rmse", "mae", "r2"])
    result_columns.insert(0, "model_idx")

    # シャッフル分割交差検証を行う
    model_idx: int = 0
    for idx, (train_data, valid_data) in enumerate(iter_kfold(load_data, n_splits=N_SPLITS)):
        # 訓練
        model = train(train_data, valid_data, model_idx=model_idx, is_save=False)

        # テスト
        rmse, mae, r2 = predict(test_data, model)
        # print(f"Score{model_idx:4d} --- RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f} ---")

        # 結果格納
        result: List[Union[float, int, str]] = deepcopy(use_columns)
        result.remove(TARGET_OBJ)
        result.extend([idx, rmse, mae, r2])
        result.insert(0, model_idx)
        result_list.append(result)
        model_idx += 1
    # 結果作成
    result_df = DataFrame(result_list, columns=result_columns)
    # 結果出力
    os.makedirs("outputs", exist_ok=True)
    result_df.to_csv(os.path.join("outputs", "All_Result_1-4.csv"), index=False)
