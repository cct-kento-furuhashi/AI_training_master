from typing import Any
import pandas as pd
import lightgbm as lgb
import pickle
from settings.params import MODEL_PATH,TRAIN_PARAMS,N_BOOST_ROUND,N_EARLY_STOPPING_ROUND
from typing import Dict,Union

# warnings.simplefilter("ignore") これも不明

def train(
    train_data:pd.DataFrame,
    test_data:pd.DataFrame,
    params: Dict[str, Union[int, float, str]] = TRAIN_PARAMS,
    model_idx: int = 0,##これの意味がわかんない。
    ) -> lgb.Booster:
    """ightGBMで訓練を実施する。


    Args:
        train_data (pd.DataFrame):訓練データ
        test_data (pd.DataFrame):テストデータ
        params:LGBハイパーパラメータ
        model_idx (int): モデルID
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


    print('Starting training...')

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=N_BOOST_ROUND,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=N_EARLY_STOPPING_ROUND,
                    verbose_eval = False,
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