import pandas
import pandas as pd
import os

horse_drop_columns = ['age', 'burden_weight', 'rider_id', 'goal_time', 'time_value', 'half_way_rank',
                      'last_time', 'odds',
                      'popular', 'horse_weight', 'remarks', 'tamer_id', 'owner_id', 'short_comment', 'date',
                      'is_down',
                      'is_senba', 'is_mesu', 'is_osu', 'distance', 'avg_velocity', 'horse_weight_dif',
                      'burden_weight_rate']

race_drop_columns = ['race_round', 'ground_status', 'date', 'where_racecourse', 'total_horse_number',
                     'horse_number_first',
                     'horse_number_second', 'horse_number_third', 'tansyo', 'hukusyo_first', 'hukusyo_second',
                     'hukusyo_third', 'wakuren', 'umaren', 'wide_1_2', 'wide_1_3', 'wide_2_3', 'umatan', 'renhuku3',
                     'rentan3', 'is_left_right_straight', 'weather_rain', 'weather_snow']


def drop_horse_data(data: pandas.DataFrame):
    data = data.drop(horse_drop_columns, axis=1)
    data.to_csv('test_horse_data.csv')


def drop_race_data(data: pandas.DataFrame):
    data = data.drop(race_drop_columns, axis=1)
    data.to_csv('test_race_data.csv')


if __name__ == '__main__':
    horse_data = pd.read_csv(os.path.join('csv', 'test_cleaned_horse_data.csv'))
    race_data = pd.read_csv(os.path.join('csv', 'test_cleaned_race_data.csv'))
    drop_horse_data(horse_data)
    drop_race_data(race_data)
