import os

INPUT_PATH = os.path.join("input_datas", "BostonHousing.csv")
TRAIN_PARAMS = {"objective": "regression", "metrics": "rmse"}
LGB_MODEL_PATH = os.path.join("static", "lgb_model.pkl")
