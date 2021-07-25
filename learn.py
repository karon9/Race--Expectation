import lightgbm as lgb
import os
import pandas as pd
from split_data import split_data


def learn(train_data, train_target, train_query):
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
    # 学習
    # train = lgb.Dataset(train_data, train_target, group=train_query)
    # valid = lgb.Dataset(val_data, val_target, reference=lgtrain, group=val_query)
    # model = lgb.train(
    #     lgbm_params,
    #     lgtrain,
    #     num_boost_round=100,
    #     valid_sets=valid,
    #     valid_names=['train', 'valid'],
    #     early_stopping_rounds=20,
    #     verbose_eval=5
    # )


if __name__ == '__main__':
    data = pd.read_csv('test_horse_data.csv').sort_values(['race_id', 'rank'])
    target_data = data['rank']
    train_data, val_data, test_data, train_target_data, val_target_data, test_target_data, train_query_data, val_query_data, test_query_data = split_data(
        data, target_data)
