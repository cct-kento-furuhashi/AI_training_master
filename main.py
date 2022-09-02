import os
from kadai1.dataShaper import shape_raw_data
from kadai1.dataloader import read_csv
from kadai1.predict import predict
from kadai1.train import trainLGB
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # ファイル読み込み
    csv_data = read_csv(os.path.join("dataset","Boston.csv"))

    # データ整形(課題1-2で実施)
    train_data,test_data = train_test_split(csv_data,test_size=56,random_state=0,shuffle=False)

    # 訓練
    model_path = trainLGB(train_data,test_data)

    # テスト
    predict(test_data,model_path)
