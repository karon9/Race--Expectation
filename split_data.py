import pandas as pd

query_data = pd.read_csv('test_query_data.csv')
split_number = int(len(query_data) / 8)
train_query_data = query_data.iloc[: split_number * 5]
val_query_data = query_data.iloc[split_number * 5:split_number * 7]
test_query_data = query_data.iloc[split_number * 7:]
first_split_number = train_query_data['horse_number'].sum()
second_split_number = first_split_number + val_query_data['horse_number'].sum()


def split_data_base_on_query(origin_data):
    train_data = origin_data.iloc[:first_split_number + 1]
    val_data = origin_data.iloc[first_split_number + 1: second_split_number + 1]
    test_data = origin_data.iloc[second_split_number + 1:]
    return train_data, val_data, test_data


def split_data(Data, Target_data):
    train_data, val_data, test_data = split_data_base_on_query(Data)
    train_target_data, val_target_data, test_target_data = split_data_base_on_query(Target_data)
    return train_data, val_data, test_data, train_target_data, val_target_data, test_target_data, train_query_data, val_query_data, test_query_data
