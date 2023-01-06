from data_loader.data_loader import read_data
from datasets.data_shaper import shape_datas
from predictor.predict_with_lgb import predict
from settings.params import INPUT_PATH
from training.train_with_optuna_lgb import train_with_optimize_param

"""
Bostonの住宅価格をLightGBMを使用して予測する
optunaのlightgbm tunerのパラメータ最適化を行い、予測精度を上げる
"""

if __name__ == "__main__":
    # ファイル読み込み
    load_data = read_data(INPUT_PATH)

    # データ整形
    train_data, valid_data, test_data = shape_datas(load_data)

    # パラメータ最適化+訓練
    model = train_with_optimize_param(train_data, valid_data)

    # テスト
    rmse, mae, r2 = predict(test_data, model)
    print(f"RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f}")
