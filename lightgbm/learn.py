import lightgbm as lgb
import pandas as pd
import os
import shap
import matplotlib.pylab as plt
from pathlib import Path

fig = plt.figure()
fig.subplots_adjust(left=0.25)
from split_data import split_data
from drop_dataset import race_id_drop
from modify_data import category_columns
from result_analysis import correct_answer_rate


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
        'random_state': 0,
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
        verbose_eval=50
    )
    return model


if __name__ == '__main__':
    # race_idにsortする。
    data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_data.csv')).sort_values(
        ['race_id', 'rank'])
    target_data = pd.Series(int(1.0 / i * 10) if i < 4 else 0 for i in data["rank"])  # 1着は10、2着は5、3着は3、4着以降は0
    data = data.drop('rank', axis=1)
    train_data, val_data, test_data, train_target, val_target, test_target, train_query, val_query, test_query = split_data(
        data, target_data)

    # race_idをdrop
    train_data = race_id_drop(train_data)
    val_data = race_id_drop(val_data)
    # test_dataのrace_idを後に紐づけるために保存
    test_race_id = test_data['race_id']
    test_data = race_id_drop(test_data)

    # カラムをカテゴリ変数に変更
    train_data = category_columns(train_data)
    val_data = category_columns(val_data)
    test_data = category_columns(test_data)

    model = learn(train_data, val_data, train_target, val_target, train_query, val_query)
    print('__________________________')
    pred = model.predict(test_data, num_iteration=model.best_iteration)

    result = pd.DataFrame({'race_id': test_race_id.values, 'predict': pred, 'result': test_target})
    tansyo, hukusyo = correct_answer_rate(result, test_query)
    print('単勝的中率 : {:.2f}%     単勝回収率 : {:.2f}%'.format(tansyo[0], tansyo[1]))
    print('複勝的中率 : {:.2f}%     複勝回収率 : {:.2f}%'.format(hukusyo[0], hukusyo[1]))
    result.to_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'result.csv'), encoding='utf_8_sig', index=False)

    # shapを使用
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)
    shap.summary_plot(shap_values=shap_values, features=train_data, feature_names=train_data.columns, plot_type='bar')