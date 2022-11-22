from datasets.data_shaper import shape_datas,select_datas
from datasets.utils import get_columns
from data_loader.data_loader import read_csv
from predictor.predict import predict
from training.train_with_lgb import train
from settings.params import INPUT_PATH,TEST_COLUMNS,ALL_COLUMNS,TARGET_OBJ

if __name__ == "__main__":

    # 全項目での確認
    load_data = read_csv(INPUT_PATH)    
    
    #--for文
    for columns in get_columns(ALL_COLUMNS,TARGET_OBJ):

        # 必要なカラムだけ抜き出す。
        selected_data = select_datas(load_data,columns)

        ## データ整形
        train_data,valid_data,test_data = shape_datas(selected_data)

        ## 訓練
        model = train(train_data,test_data)

        ## テスト
        rmse,mae,r2 = predict(test_data,model)
        print(f"RMSE: {rmse:.2f} MAE: {mae:.2f} R2: {r2:.2f}")
    #--for文範囲
    
