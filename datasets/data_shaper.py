import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

def shape_raw_data(raw_data:pd.DataFrame)->pd.DataFrame:
    """_summary_
        現時点(220912)では1行目を削る関数。
    Args:
        raw_data (pd.DataFrame): データフレーム型のデータ

    Returns:
        pd.DataFrame: 整形後のデータ
    """
    shaped_data = raw_data.drop(0, axis=1)
    return shaped_data

def shape_datas(df:pd.DataFrame)->pd.DataFrame:
    """_summary_
        現時点(220912)では1行目を削る関数。
    Args:
        raw_data (pd.DataFrame): データフレーム型のデータ

    Returns:
        pd.DataFrame: 整形後のデータ
    """
    train_valid_df, test_df = train_test_split(df, test_size=56, shuffle=False)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, shuffle=False)
    return train_df, valid_df, test_df


def select_data(raw_data:pd.DataFrame,select_name:list,target_name:str)->pd.DataFrame:
    """_summary_
    select_nameで送られてきたカラム名がデータフレームにあるか確認、ある項目だけを返す。
    Args:
        raw_data (pd.DataFrame): 整形前のデータ
        select_name (list): 残しておきたいカラム名
        target_name (str): 目的変数

    Returns:
        pd.DataFrame: 選択した説明変数+目的変数で構成されたデータフレーム
    """
    selected_name = select_name.copy()
    selected_name.append(target_name)
    selected_data = raw_data.loc[:,selected_name]
    return selected_data

def combination_brute_forse(explan_values:list,combination_num:int)->list:
    """_summary_
        explan_valuesで送られてきたリストをcombination_numの個数取り出す際の全ての組み合わせ（重複無し）を作成し、返す。
    Args:
        explan_values (list): 元のリスト
        combination_num (int): 組み合わせ数

    Returns:
        list: 全ての組み合わせ数
    """
    all_comb = itertools.permutations(explan_values,combination_num)
    for x in all_comb:
        print(x)

    return all_comb
