from settings.params import INPUT_PATH
from data_loader.data_loader import read_csv
from datasets.data_shaper import shape_datas
from training.train_with_lgb import train
from predictor.predict import predict
from optimizer.optimize import optimize_param


if __name__ == "__main__":

    # 全項目での確認
    load_data = read_csv(INPUT_PATH)    
    
    ## データ整形
    train_data,valid_data,test_data = shape_datas(load_data)
    
    ## パラメータ最適化
    params = optimize_param(train_data,valid_data,test_data)

    ## 訓練
    model = train(train_data,test_data,params=params)
    
    ## テスト
    rmse,mae,r2 = predict(test_data,model)
    print(f"RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f}")
