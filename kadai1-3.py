import os
import random
import pprint
from turtle import pd
import pandas as pd 
from unittest import result
from kadai.dataShaper import shape_raw_data
from kadai.dataShaper import select_data
from kadai.dataShaper import combination_brute_forse
from kadai.dataloader import read_csv
from kadai.predict import predict
from kadai.train import trainLGB
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # # 全項目での確認
    ## ファイル読み込み
    csv_data = read_csv(os.path.join("input_data","Boston.csv"))    

    # 結果を入れるDF
    result_list_all = []
    ## 使用するカラム
    best_columns = ["crim","rm","dis","age","lstat"]
    ## データの選択
    processed_data = select_data(csv_data,best_columns,"medv")
    ## データ整形
    train_data,test_data = train_test_split(processed_data,test_size=56,random_state=0,shuffle=False)
    ## 訓練
    model_path = trainLGB(train_data,test_data)
    ## テスト
    predict_result,importance_result = predict(test_data,model_path)
    ## 結果を格納 
    ### 辞書型
    # result_dict = {'best_'+'_'.join(["crim","rm","dis","age","lstat"]):predict_result}
    # importance_sum = {'best_'+'_'.join(["crim","rm","dis","age","lstat"]):(importance_base.T.loc[:,best_columns].T['importance'].sum())}
    ### リスト
    result_list = ['best','crim:' +str(importance_result['crim']),
                          'rm:'   +str(importance_result['rm']),
                          'dis:'  +str(importance_result['dis']),
                          'age:'  +str(importance_result['age']),
                          'lstat:'+str(importance_result['lstat']),
                          predict_result,
                          importance_result.T.loc[:,best_columns].T['importance'].sum()]
    result_list_all.append(result_list)

    # 任意の5項目を使用し推論
    for i in range(10):
        ## 使用するカラム
        random_columns = random.sample(list(csv_data.drop("medv",axis=1).columns.values),k=5)
        ## データ選択
        processed_data = select_data(csv_data,random_columns,"medv")
        ## データ整形
        train_data,test_data = train_test_split(processed_data,test_size=56,random_state=0,shuffle=False)
        ## 訓練
        model_path = trainLGB(train_data,test_data)
        ## テスト
        predict_result,importance_result = predict(test_data,model_path)
        ## 結果を格納
        ### 辞書型
        # result_dict['rnd'+str(i)+'_'+'_'.join(random_columns)] = predict_result
        # importance_sum['rnd'+str(i)+'_'+'_'.join(random_columns)] = importance_base.T.loc[:,random_columns].T['importance'].sum()
        ### リスト
        result_list = ['rnd'+str(i),
                        random_columns[0]+':'+str(imp_base_dict[random_columns[0]]),
                        random_columns[1]+':'+str(imp_base_dict[random_columns[1]]),
                        random_columns[2]+':'+str(imp_base_dict[random_columns[2]]),
                        random_columns[3]+':'+str(imp_base_dict[random_columns[3]]),
                        random_columns[4]+':'+str(imp_base_dict[random_columns[4]]),
                        predict_result,importance_base.T.loc[:,random_columns].T['importance'].sum()]
                        
        result_list_all.append(result_list)

    # 結果表示
    print('-------\n')
    result_df = pd.DataFrame(result_list_all,columns=['Name','n=1','n=2','n=3','n=4','n=5','RMSE','importance'])
    print(result_df)
    print('-------\n')

