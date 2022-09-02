import csv
from typing import Any
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import os

def trainLGB(train_data:pd.DataFrame,test_data:pd.DataFrame) -> Any:
    """ightGBMで訓練を実施する。


    Args:
        train_data (pd.DataFrame):訓練データ
        test_data (pd.DataFrame):テストデータ
   
    Returns:
        Any: 未定
    """

    # Dataframe型の中から、Xとyを抽出する
    # X <= CSVのMEDV以外の項目
    # y <= MEDVだけ
    model_path = os.path.join("kadai1","models","model.npy")

    y_train = train_data["medv"].values
    X_train = train_data.drop("medv",axis=1).values

    y_test = test_data["medv"].values
    X_test = test_data.drop("medv",axis=1).values


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train,y_train)
    lgb_eval = lgb.Dataset(X_test,y_test)

    print("--- lgb_dataset created ---")


    # specify your configurations as a dict
    params = {
        'objective': 'regression',
        'metrics': 'rmse',
    }

    print('Starting training...')

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100
                    )

    print('Saving model...')

    with open(model_path,"wb") as f:
        pickle.dump(gbm,f)

    return model_path