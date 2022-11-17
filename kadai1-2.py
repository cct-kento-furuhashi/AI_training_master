from data_loader.data_loader import read_data
from datasets.data_shaper import shape_datas
from optimizer.optimize_param import optimize_param
from predictor.predict_with_lgb import predict
from settings.params import INPUT_PATH
from training.train_with_lgb import train

"""
Bostonの住宅価格をLightGBMを使用して予測する
パラメータ最適化を行い、予測精度を上げる
"""

if __name__ == "__main__":
    # ファイル読み込み
    load_data = read_data(INPUT_PATH)

    # データ整形
    train_data, valid_data, test_data = shape_datas(load_data)

    # パラメータ最適化
    params = optimize_param(train_data, valid_data, test_data)

    # 訓練
    model = train(train_data, valid_data, params=params)

    # テスト
    rmse, mae, r2 = predict(test_data, model)
    print(f"RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f}")
