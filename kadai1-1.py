import os
import random
import pprint
from turtle import pd
import pandas as pd 
from unittest import result
from datasets.data_shaper import shape_datas
from data_loader.data_loader import read_csv
from predictor.predict import predict
from training.train import trainLGB
from settings.params import INPUT_PATH


if __name__ == "__main__":

    # 全項目での確認
    load_data = read_csv(INPUT_PATH)    
    ## データ整形
    train_data,valid_data,test_data = shape_datas(load_data)
    ## 訓練
    model = trainLGB(train_data,test_data)
    ## テスト
    rmse,mae,r2 = predict(test_data,model)
    print(f"RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f}")