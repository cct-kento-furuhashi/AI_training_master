import os
from typing import Dict, Union

INPUT_PATH = os.path.join("input_data","Boston.csv") #pathはparamsにすべて書きたい。
TRAIN_PARAMS: Dict[str, Union[int, float, str]] = {
    "objective": "regression",
    "metrics": "rmse",
    "verbosity": -1,
}  # 訓練パラメータ
MODEL_PATH = os.path.join("models","model.npy")
TARGET_OBJ = "medv" #目的変数名（なるほどparamsのコメントは特に丁寧に書いてもいいかも。
SEED = 831 #801
N_BOOST_ROUND = 10000 #ループ数
N_EARLY_STOPPING_ROUND = 100 #EARLY STOPPOINGのラウンド数
PARAM_PATH = os.path.join("params.json")
TRIAL_NUM = 100

ALL_COLUMNS = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv"]
TEST_COLUMNS = ["crim","tax","ptratio","black","lstat","medv"]