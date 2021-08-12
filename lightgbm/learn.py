import time

import lightgbm as lgb
import pandas as pd
import os
import shap
import matplotlib.pylab as plt
from pathlib import Path
from optuna.integration import lightgbm as LGB_optuna
import pickle
import datetime
import numpy as np

dt_now = datetime.datetime.now()

fig = plt.figure()
fig.subplots_adjust(left=0.25)
from split_data import split_data
from drop_dataset import race_id_drop
from modify_data import category_columns
from result_analysis import correct_answer_rate


def learn(Train_data, Val_data, Train_target, Val_target, Train_query, Val_query, label_value):
    Train_query = pd.Series(Train_query['horse_number'])
    Val_query = pd.Series(Val_query['horse_number'])
    # 学習
    lgb_train = lgb.Dataset(Train_data, Train_target, group=Train_query)
    lgb_valid = lgb.Dataset(Val_data, Val_target, group=Val_query, reference=lgb_train)
    if args.optuna:
        print('use optuna !!')
        time.sleep(1)
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 2, 3, 4, 5],
            'feature_pre_filter': False,
            'force_col_wise': True,
            'label_gain': label_value
        }
        best = LGB_optuna.train(param, lgb_train, valid_sets=lgb_valid, verbose_eval=50)
        print(best.params)
        model = lgb.train(best.params, lgb_train, verbose_eval=50)
    else:
        # lightGBMのパラメータ設定
        params = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [1, 2, 3, 4, 5],
                  'feature_pre_filter': False,
                  'force_col_wise': True, 'lambda_l1': 0.0, 'lambda_l2': 0.0, 'num_leaves': 4, 'feature_fraction': 0.4,
                  'bagging_fraction': 0.9725813736058717, 'bagging_freq': 2, 'min_child_samples': 20,
                  'num_iterations': 500, 'early_stopping_round': None,
                  'label_gain': label_value}
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=300,
            valid_sets=lgb_valid,
            valid_names=['train', 'valid'],
            verbose_eval=50
        )
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna", help="using optuna", action="store_true")
    args = parser.parse_args()

    # race_idにsortする。
    data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'learn_data.csv'))

    # target_data = data['goal_time_dif'].astype(int)
    # target_data = target_data.apply(lambda x: 30 if x > 30 else x)
    # target_data = data['time_value']
    # target_data = target_data.apply(lambda x: 0 if x < 0 else x)
    # label_value = sorted(list(target_data.unique()))
    # label_value = list(map(int, label_value))
    # target_data = pd.concat([target_data, data['rank']], axis=1)
    target_data = data['rank'].astype(str)
    target_data = target_data.apply(lambda x: int(3) if x == '1' else x)
    target_data = target_data.apply(lambda x: int(2) if x == '2' else x)
    target_data = target_data.apply(lambda x: int(1) if x == '3' else x)
    target_data = target_data.apply(lambda x: int(0) if type(x) == str else x)
    label_value = sorted(list(target_data.unique()))
    label_value = list(map(str, label_value))

    data = data.drop(['goal_time_dif', 'rank', 'time_value'], axis=1)
    # data['half_way_rank'] = data['half_way_rank'].astype(int).astype('category')
    # data = data.drop(['half_way_rank'], axis=1)

    # data = data.drop(columns='rank', axis=1)
    train_data, val_data, test_data, train_target, val_target, test_target, train_query, val_query, test_query = split_data(
        data, target_data)

    # race_id,dateをdrop
    # test_dataのrace_idを後に紐づけるために保存
    test_tm_data = pd.concat([test_data['race_id'], test_data['date']], axis=1)
    test_data = race_id_drop(test_data)
    train_data = race_id_drop(train_data)
    val_data = race_id_drop(val_data)

    # カラムをカテゴリ変数に変更
    train_data = category_columns(train_data)
    val_data = category_columns(val_data)
    test_data = category_columns(test_data)

    model = learn(train_data, val_data, train_target, val_target, train_query,
                  val_query, label_value)
    print('__________________________')
    file = f'{dt_now.year}-{dt_now.month}-{dt_now.day}.pkl'
    pickle.dump(model, open(os.path.join(Path(os.getcwd()).parent, 'model', file), 'wb'))
    test_data.to_csv(os.path.join(Path(os.getcwd()).parent, 'test.csv'), encoding='utf_8_sig', index=False)
    # test_target.to_csv(os.path.join(Path(os.getcwd()).parent, 'test_target_today.csv'), encoding='utf_8_sig', index=False)

    pred = model.predict(test_data, num_iteration=model.best_iteration)
    test_data = pd.concat([test_data, test_tm_data], axis=1)

    result = pd.DataFrame(
        {'date': test_data['date'], 'horse_number': test_data['horse_number'], 'race_id': test_data['race_id'].values,
         'predict': pred, 'result': test_target})
    tansyo, hukusyo = correct_answer_rate(result, test_query)
    print('単勝的中率 : {:.2f}%     単勝回収率 : {:.2f}%'.format(tansyo[0], tansyo[1]))
    print('複勝的中率 : {:.2f}%     複勝回収率 : {:.2f}%'.format(hukusyo[0], hukusyo[1]))
    result.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'result.csv'), encoding='utf_8_sig', index=False)

    # shapを使用
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)
    shap.summary_plot(shap_values=shap_values, features=train_data, feature_names=train_data.columns, plot_type='bar')
