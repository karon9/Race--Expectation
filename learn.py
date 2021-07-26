import lightgbm as lgb
import os
import pandas as pd
from split_data import split_data
from drop_dataset import unnamed_race_id_drop


def learn(Train_data, Val_data, Train_target, Val_target, Train_query, Val_query):
    # lightGBMのパラメータ設定
    max_position = 20
    lgbm_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'lambdarank_truncation_level': 10,
        'ndcg_eval_at': [1, 2, 3],
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'random_state': 777,
    }
    Train_query = pd.Series(Train_query['horse_number'])
    Val_query = pd.Series(Val_query['horse_number'])
    # 学習
    lgb_train = lgb.Dataset(Train_data, Train_target, group=Train_query)
    lgb_valid = lgb.Dataset(Val_data, Val_target, group=Val_query, reference=lgb_train)
    model = lgb.train(
        params=lgbm_params,
        train_set=lgb_train,
        num_boost_round=100,
        valid_sets=lgb_valid,
        valid_names=['train', 'valid'],
        early_stopping_rounds=20,
        verbose_eval=5
    )


if __name__ == '__main__':
    # race_idにsortする。
    data = pd.read_csv('test_horse_data.csv').sort_values(['race_id', 'rank'])
    target_data = data['rank']
    data = data.drop('rank', axis=1)
    train_data, val_data, test_data, train_target, val_target, test_target, train_query, val_query, test_query = split_data(
        data, target_data)
    train_data = unnamed_race_id_drop(train_data)
    val_data = unnamed_race_id_drop(val_data)
    test_data = unnamed_race_id_drop(test_data)
    learn(train_data, val_data, train_target, val_target, train_query, val_query)
