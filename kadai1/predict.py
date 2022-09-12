from sklearn.metrics import mean_squared_error
import pickle
import lightgbm
import pandas as pd 

def predict(test_data:pd.DataFrame , model_path:str):

    y_test = test_data["medv"].values
    X_test = test_data.drop("medv",axis=1).values

    with open(model_path,"rb") as f:
        gbm =  pickle.load(f)

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'The RMSE of prediction is: {rmse_test}')

    ## featureimportance
    importance = pd.DataFrame(gbm.feature_importance(),index=test_data.drop("medv",axis=1).columns,columns=['importance'])
    
    return rmse_test,importance