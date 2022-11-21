from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import lightgbm as lgb
import pandas as pd 
import numpy as np
from typing import Tuple

def predict(test_data:pd.DataFrame , model: lgb.Booster) -> Tuple[float, float, float]:
    """predict 予測
    Args:
        test_df (pd.DataFrame): テストデータ
        model (lgb.Booster): モデル
    Returns:
        Tuple[float, float, float]: RMSE(二乗平均平方根誤差), MAE(平均絶対誤差), R2スコア
    """
    y_test = test_data["medv"].values
    X_test = test_data.drop("medv",axis=1).values

    print('Starting predicting...')
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    r2   = r2_score(y_test,y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    ## featureimportance
    ##importance = pd.DataFrame(model.feature_importance(),index=test_data.drop("medv",axis=1).columns,columns=['importance'])
    
    return rmse,mae,r2