import pandas
import pandas as pd
import os

horse_drop_columns = ['goal_time', 'half_way_rank', 'time_value', 'last_time', 'remarks', 'odds', 'distance', 'popular',
                      'horse_weight', 'tamer_id', 'owner_id', 'short_comment', 'date', 'avg_velocity',
                      'burden_weight_rate', ]

race_drop_columns = ['race_round', 'ground_status', 'date', 'horse_number_first',
                     'horse_number_second', 'horse_number_third', 'tansyo', 'hukusyo_first', 'hukusyo_second',
                     'hukusyo_third', 'wakuren', 'umaren', 'wide_1_2', 'wide_1_3', 'wide_2_3', 'umatan', 'renhuku3',
                     'rentan3']

odds = ['tansyo', 'hukusyo_first', 'hukusyo_second',
        'hukusyo_third', 'wakuren', 'umaren', 'wide_1_2', 'wide_1_3', 'wide_2_3', 'umatan', 'renhuku3',
        'rentan3']


def drop_horse_data(df: pandas.DataFrame):
    df = df.drop(horse_drop_columns, axis=1)
    df.to_csv(os.path.join(os.getcwd(), 'test_csv', 'test_horse_data.csv'), index=False)


def drop_race_data(df: pandas.DataFrame):
    df = df.drop(race_drop_columns, axis=1)
    df.to_csv(os.path.join(os.getcwd(), 'test_csv', 'test_race_data.csv'), index=False)


def race_id_drop(df):
    df = df.drop('race_id', axis=1)
    return df


def drop_odds(df):
    df = df.drop('')


if __name__ == '__main__':
    horse_data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_cleaned_horse_data.csv'))
    race_data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_cleaned_race_data.csv'))
    drop_horse_data(horse_data)
    drop_race_data(race_data)
