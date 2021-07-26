from sklearn import preprocessing
import pandas as pd
import os
import numpy as np


def category_columns(df):
    number_column = ['horse_number', 'burden_weight', 'horse_weight_dif', 'total_horse_number', 'baba_index',
                     'distance']
    for column in df.columns:
        if column in number_column:
            df[column] = df[column].astype(np.int32)
        else:
            df[column] = df[column].astype('category')
    return df


if __name__ == '__main__':
    horse_data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_horse_data.csv'))
    race_data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_race_data.csv'))
    data = pd.merge(horse_data, race_data, on='race_id')
    data.to_csv(os.path.join(os.getcwd(), 'test_csv', 'test_data.csv'), encoding='utf_8_sig', index=False)
