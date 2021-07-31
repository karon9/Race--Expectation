import pandas as pd
import os
from pathlib import Path

query_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'learn_query_data.csv'))
split_number = int(len(query_data) / 8)
train_query_data = query_data.iloc[: split_number * 4]
val_query_data = query_data.iloc[split_number * 4:split_number * 6]
test_query_data = query_data.iloc[split_number * 6:]
first_split_number = train_query_data['horse_number'].sum()
second_split_number = first_split_number + val_query_data['horse_number'].sum()


def split_data_base_on_query(origin_data, target=False):
    train_data = origin_data.iloc[:first_split_number]
    val_data = origin_data.iloc[first_split_number: second_split_number]
    test_data = origin_data.iloc[second_split_number:]
    if target:
        test_data = test_replace(test_data)
        val_data = test_replace(val_data)
    return train_data, val_data, test_data


def split_data(Data, Target_data):
    # traget のカラムを変更したいならtarget = True
    train_data, val_data, test_data = split_data_base_on_query(Data, target=True)
    train_target_data, val_target_data, test_target_data = split_data_base_on_query(Target_data, target=False)
    return train_data, val_data, test_data, train_target_data, val_target_data, test_target_data, train_query_data, val_query_data, test_query_data


def test_replace(df):
    df['remarks'] = pd.DataFrame(['' for i in range(df.index.size)]).astype('category')
    return df
