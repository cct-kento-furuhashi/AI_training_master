import pandas as pd

def shape_raw_data(raw_data:pd.DataFrame)->pd.DataFrame:
    ## 1行目をけづる
    shaped_data = raw_data.drop(0, axis=1)

    
    return shaped_data

