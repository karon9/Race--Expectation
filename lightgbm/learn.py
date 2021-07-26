import lightgbm as lgb
import pandas as pd
import os
import shap
from race_html.split_data import split_data
from drop_dataset import race_id_drop
from modify_data import category_columns


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
    return model


if __name__ == '__main__':
    # race_idにsortする。
    data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_data.csv')).sort_values(['race_id', 'rank'])
    target_data = pd.Series(int(1.0 / i * 10) if i < 4 else 0 for i in data["rank"])  # 1着は10、2着は5、3着は3、4着以降は0
    data = data.drop('rank', axis=1)
    train_data, val_data, test_data, train_target, val_target, test_target, train_query, val_query, test_query = split_data(
        data, target_data)
    train_data = race_id_drop(train_data)
    val_data = race_id_drop(val_data)
    test_data = race_id_drop(test_data)

    # カラムをカテゴリ変数に変更
    train_data = category_columns(train_data)
    val_data = category_columns(val_data)
    test_data = category_columns(test_data)

    model = learn(train_data, val_data, train_target, val_target, train_query, val_query)
    print('__________________________')
    pred = model.predict(test_data, num_iteration=model.best_iteration)

    result = pd.DataFrame({'予想': pred, '実際': test_target})
    result.to_csv(os.path.join(os.getcwd(), 'test_csv', 'result.csv'), encoding='utf_8_sig', index=False)

    # shapを使用
    # explainer = shap.TreeExplainer(model, data=train_data)
    # tr_x_shap_values = explainer.shap_values(train_data)
    # shap.summary_plot(shap_values=tr_x_shap_values, features=train_data, feature_names=train_data.columns)
    print(result)
