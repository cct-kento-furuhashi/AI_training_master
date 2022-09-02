from data_loader.data_loader import read_data
from datasets.data_shaper import shape_datas
from predictor.predict_with_lgb import predict
from settings.params import INPUT_PATH
from training.train_with_lgb import train

if __name__ == "__main__":
    # ファイル読み込み
    load_data = read_data(INPUT_PATH)

    # データ整形
    train_data, valid_data, test_data = shape_datas(load_data)

    # 訓練
    model = train(train_data, valid_data)

    # テスト
    predict(test_data, model)
