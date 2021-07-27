import pandas as pd
import os
from pathlib import Path
import numpy as np


def correct_answer_rate(df, query_data):
    odds_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_odds_data.csv'))
    df = pd.merge(df, odds_data, on='race_id')
    # resultの並び順を降順にして、同じpredictの時に悪い方になるようにした
    df = df.sort_values(["race_id", "predict", "result"], ascending=[True, False, True]).reset_index(drop=True)
    tansyo_rate = tansyo(df, query_data)
    hukusyo_rate = hukusyo(df, query_data)
    return tansyo_rate, hukusyo_rate


def tansyo(df, query_df, first_index=0):
    hit_count = 0
    race_count = 0
    dividen = 0
    for query_num in list(query_df.values.flatten().tolist()):
        if df['result'][first_index] == 10:
            hit_count += 1
            dividen += df['tansyo'][first_index].astype(np.int32)
        race_count += 1
        first_index += query_num
    return hit_count / race_count * 100, dividen / race_count


def hukusyo(df, query_df, first_index=0, second_index=1, third_index=2):
    hit_count = 0
    race_count = 0
    dividen = 0
    for query_num in list(query_df.values.flatten().tolist()):
        if df['result'][first_index] == 10:
            hit_count += 1
            dividen += df['hukusyo_first'][first_index].astype(np.int32)
        if df['result'][first_index] == 5:
            hit_count += 1
            dividen += df['hukusyo_second'][first_index].astype(np.int32)
        if df['result'][first_index] == 3:
            hit_count += 1
            dividen += df['hukusyo_third'][first_index].astype(np.int32)
        race_count += 1
        first_index += query_num
    return hit_count / race_count * 100, dividen / race_count


if __name__ == '__main__':
    result_df = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'result.csv'))
    test_query_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_query_data.csv'))
    test_query_data = test_query_data[9978:13309]
    correct_answer_rate(result_df, test_query_data)
