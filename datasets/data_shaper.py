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

def select_datas(df:pd.DataFrame,colums:list)->pd.DataFrame:
    """_summary_
        特定のカラムを取り出して、データフレームで返す
    Args:
        df (pd.DataFrame): 元データ
        colums (list): 取り出すカラム

    Returns:
        pd.DataFrame: 必要なカラムだけのデータ
    """
    return df[colums]

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
