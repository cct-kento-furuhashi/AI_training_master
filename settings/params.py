import os

INPUT_PATH = os.path.join("input_datas", "BostonHousing.csv")  # 入力データ
TRAIN_PARAMS = {"objective": "regression", "metrics": "rmse", "verbosity": -1}  # 訓練パラメータ
os.makedirs("static", exist_ok=True)  # モデル格納先作成
LGB_MODEL_PATH = os.path.join("static", "lgb_model_{:03d}.pkl")  # モデルパス
TARGET_OBJ = "medv"  # 目的変数名
N_SPLITS = 5  # K-Fold分割数
USE_COLUMN_NUM = 5  # 使用する列の数
N_BOOST_ROUND = 1000  # ループ数
N_EARLY_STOPPING_ROUND = 100  # EARLY STOPPINGのラウンド数
