from typing import Any
import pandas as pd
import lightgbm as lgb
import pickle
from settings.params import MODEL_PATH

def trainLGB(train_data:pd.DataFrame,test_data:pd.DataFrame) -> lgb.Booster:
    """ightGBMで訓練を実施する。


    Args:
        train_data (pd.DataFrame):訓練データ
        test_data (pd.DataFrame):テストデータ
   
    Returns:
        lgb.Booster: 訓練済みモデル
    """

    # Dataframe型の中から、Xとyを抽出する
    # X <= CSVのMEDV以外の項目
    # y <= MEDVだけ

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
        'lerning_rate': 0.1,
        'num_iterations': 100,
        'num_leaves':31,
        'max_depth':-1,
        'verbosity':-1
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
  
    with open(MODEL_PATH,"wb") as f:
        pickle.dump(gbm,f)

    return gbm

# def objective(trial):
#     model_path = os.path.join("kadai","models","model.npy")

#     y_train = train_data["medv"].values
#     X_train = train_data.drop("medv",axis=1).values

#     y_test = test_data["medv"].values
#     X_test = test_data.drop("medv",axis=1).values


#     # create dataset for lightgbm
#     lgb_train = lgb.Dataset(X_train,y_train)
#     lgb_eval = lgb.Dataset(X_test,y_test)

#     param = {
#         'objective': 'regression',
#         'metrics': 'rmse',
#         'lerning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
#         'num_iterations': trial.suggest_int('num_iterations', 10, 1000),
#         'num_leaves':trial.suggest_int('num_leaves', 10, 1000),
#         'max_depth':-1,
#         'verbosity':-1
#     }
#     train_xy = lgb.Dataset(train_x, train_y)
#     val_xy = lgb.Dataset(test_x, test_y, reference=train_xy)

#     # train
#     gbm = lgb.train(param,
#                     lgb_train,
#                     num_boost_round=1000,
#                     valid_sets=lgb_eval,
#                     early_stopping_rounds=100
#                     )
    
#     pred_proba = gbm.predict(test_x)
#     pred = np.argmax(pred_proba, axis=1)
    
#     acc = sklearn.metrics.accuracy_score(test_y, pred)
#     return acc


# def optunaTrainLGB(train,test):
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=100)