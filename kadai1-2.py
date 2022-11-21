import os
import random
import pprint
from turtle import pd
import pandas as pd 
from unittest import result
from kadai.dataShaper import shape_raw_data
from kadai.dataShaper import select_data
from kadai.dataShaper import combination_brute_forse
from kadai.dataloader import read_csv
from kadai.predict import predict
from kadai.train import trainLGB
from sklearn.model_selection import train_test_split
from kadai.settings.params import INPUT_PATH



if __name__ == "__main__":

    # 全項目での確認
    ## ファイル読み込み
    csv_data = read_csv(INPUT_PATH)    
    ## データ整形
    ##train_data,test_data = train_test_split(csv_data,test_size=56,random_state=0,shuffle=False)
    train_data,valid_data,test_data = shape_datas(csv_data)


    ## パラメータ最適化
    params = optimize_param(train_data, valid_data, test_data)

    ## 訓練
    model_path = trainLGB(train_data,test_data)
    ## テスト
    predict_result,importance_result = predict(test_data,model_path)
