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

dt_now = datetime.datetime.now()

fig = plt.figure()
fig.subplots_adjust(left=0.25)
from split_data import split_data
from drop_dataset import race_id_drop
from modify_data import category_columns
from result_analysis import correct_answer_rate


def learn(Train_data, Val_data, Train_target, Val_target, Train_query, Val_query):
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
            'ndcg_eval_at': [1, 2, 3],
            'feature_pre_filter': False,
            'force_col_wise': True,
        }
        best = LGB_optuna.train(param, lgb_train, valid_sets=lgb_valid, verbose_eval=50)
        print(best.params)
        Train_data = pd.concat([Train_data, Val_data], axis=0)
        Train_query = pd.concat([Train_query, Val_query], axis=0)
        Train_target = pd.concat([Train_target, Val_target], axis=0)
        lgb_train = lgb.Dataset(Train_data, Train_target, group=Train_query)
        model = lgb.train(best.prams, lgb_train, verbose_eval=50)
        pickle.dump(best.prams, open(os.path.join(Path(os.getcwd()).parent, 'params.csv'), 'wb'))
    else:
        with open(os.path.join(Path(os.getcwd()).parent, 'params.csv'), mode='rb') as f:
            print('load model...')
            prams = pickle.load(f)
        # lightGBMのパラメータ設定
        lgbm_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'lambdarank_truncation_level': 10,
            'ndcg_eval_at': [1, 2, 3],
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'random_state': 0,
        }
        model = lgb.train(
            params=prams,
            train_set=lgb_train,
            verbose_eval=50
        )
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna", help="using optuna", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'learn_data.csv'))

    # target_data = pd.Series(int(10 / i) if i < 6 else 0 for i in data["rank"])  # 1着は10、2着は5、3着は3、4着以降は0
    target_data = data['goal_time_dif'].astype(int)
    target_data = target_data.apply(lambda x: 30 if x > 30 else x)
    target_data = pd.concat([target_data, data['rank']], axis=1)
    data = data.drop(['goal_time_dif', 'rank'], axis=1)
    data['half_way_rank'] = data['half_way_rank'].astype(int).astype('category')

    train_data, val_data, test_data, train_target, val_target, test_target, train_query, val_query, test_query = split_data(
        data, target_data)
    train_data = pd.concat([train_data, val_data], axis=0)
    train_target = pd.concat([train_target, val_target], axis=0)

    # data = data.drop(columns='rank', axis=1)
    train_query = pd.concat([train_query, val_query], axis=0)

    # race_idをdrop
    train_data = race_id_drop(train_data)
    test_data = race_id_drop(test_data)
    # test_dataのrace_idを後に紐づけるために保存

    # カラムをカテゴリ変数に変更
    train_data = category_columns(train_data)
    test_data = category_columns(test_data)

    file = f'{dt_now.year}-{dt_now.month}-{dt_now.day}.pkl'
    file = os.path.join(Path(os.getcwd()).parent, 'model', file)
    if not os.path.exists(file):
        model = learn(train_data, test_data, train_target['goal_time_dif'], test_target['goal_time_dif'], train_query,
                      test_query)
    else:
        with open(file, mode='rb') as f:
            print('load model...')
            model = pickle.load(f)
    print('__________________________')

    pickle.dump(model, open(file, 'wb'))

    pre_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'predict.csv'))
    # test_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test.csv'))
    pre_data = category_columns(pre_data)
    # test_target = pd.read_csv(os.path.join((Path(os.getcwd()).parent), 'test_target.csv'))
    pre_query = pd.DataFrame(pre_data.groupby('date')['horse_number'].count()).reset_index(drop=True)

    # test_dataのrace_idを後に紐づけるために保存
    pre_race_id = pre_data['race_id']
    pre_data = race_id_drop(pre_data)

    pred = model.predict(pre_data, num_iteration=model.best_iteration)

    result = pd.DataFrame(
        {'race_id': pre_race_id.values, 'number': pre_data['horse_number'], 'predict': pred})
    result.to_csv(os.path.join(Path(os.getcwd()).parent, 'result.csv'), encoding='utf_8_sig', index=False)
    # shapを使用
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(train_data)
    # shap.summary_plot(shap_values=shap_values, features=train_data, feature_names=train_data.columns)
