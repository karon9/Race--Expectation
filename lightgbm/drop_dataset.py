import pandas
import pandas as pd
import os
from pathlib import Path

# 必要ないかもしれないデータ
after_add_horse_columns = ['age', 'burden_weight', 'half_way_rank', 'date_month', 'is_mesu', 'is_osu',
                           'horse_weight_dif']
after_add_race_columns = ['total_horse_number', 'is_left_right_straight']

odds = ['tansyo', 'hukusyo_first', 'hukusyo_second',
        'hukusyo_third', 'wakuren', 'umaren', 'wide_1_2', 'wide_1_3', 'wide_2_3', 'umatan', 'renhuku3',
        'rentan3']

horse_drop_columns = ['goal_time', 'time_value', 'last_time', 'distance',
                      'horse_weight', 'tamer_id', 'owner_id', 'short_comment', 'avg_velocity',
                      'burden_weight_rate', 'is_senba', 'is_down']

race_drop_columns = ['race_round', 'ground_status', 'horse_number_first',
                     'horse_number_second', 'horse_number_third', 'weather_rain', 'weather_snow'] + odds


def drop_horse_data(df: pandas.DataFrame):
    df = df.drop(horse_drop_columns, axis=1)
    df = df.drop(after_add_horse_columns, axis=1)
    df = df.drop(['odds', 'popular', 'remarks'], axis=1)  # 過学習を起こしてる可能性
    df.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'horse_data.csv'), index=False)
    return df


def drop_race_data(df: pandas.DataFrame):
    df_odds = pd.concat([df['race_id'], df[odds]], axis=1)
    df_odds.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'odds_data.csv'), index=False)
    df = df.drop(race_drop_columns + after_add_race_columns, axis=1)
    df.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'race_data.csv'), index=False)
    return df


def race_id_drop(df):
    df = df.drop('race_id', axis=1)
    return df


def drop_odds(df):
    df = df.drop('')


if __name__ == '__main__':
    horse_data = pd.read_csv(os.path.join(os.getcwd(), 'csv', 'cleaned_horse_data.csv'))
    race_data = pd.read_csv(os.path.join(os.getcwd(), 'csv', 'cleaned_race_data.csv'))
    drop_horse_data(horse_data)
    drop_race_data(race_data)
