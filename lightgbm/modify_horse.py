import pandas
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


def shift_df(df_horse: pandas.DataFrame):
    df_race = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'cleaned_race_data.csv'))
    df_race = pd.concat([df_race['race_id'], df_race['date']], axis=1)
    df = pd.merge(df_horse, df_race, on='race_id')
    df = df.sort_values(['horse_id', 'date'], ascending=[True, True]).reset_index(drop=True)
    query_dataset = df['horse_id'].value_counts().reset_index().sort_values('index').rename(
        columns={'index': 'horse_id', 'horse_id': 'count'})
    horse_id_count = list(map(int, query_dataset['horse_id'].unique()))
    firstLoop = True
    print("shift horse race time_value..")
    for horse_id in tqdm(horse_id_count):
        df_tm_horse_ = df[df['horse_id'] == horse_id]
        df_tm_horse_1 = df_tm_horse_['time_value'].shift(1).rename('time_value_1')
        df_tm_horse_2 = df_tm_horse_['time_value'].shift(2).rename('time_value_2')
        df_tm_horse_3 = df_tm_horse_['time_value'].shift(3).rename('time_value_3')
        df_tm_horse = pd.concat([df_tm_horse_, df_tm_horse_1, df_tm_horse_2, df_tm_horse_3], axis=1)
        if firstLoop:
            df_tm = df_tm_horse
            firstLoop = False
        else:
            df_tm = pd.concat([df_tm, df_tm_horse], axis=0)
    df = df_tm.drop('date', axis=1)
    df.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'horse_shift.csv'), encoding='utf_8_sig', index=False)
    return df


def half_way_rank_ave(df_horse: pd.DataFrame):
    df_race = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'cleaned_race_data.csv'))
    df_race = pd.concat([df_race['race_id'], df_race['date']], axis=1)
    df = pd.merge(df_horse, df_race, on='race_id')
    df = df.sort_values(['horse_id', 'date'], ascending=[True, True]).reset_index(drop=True)
    query_dataset = df['horse_id'].value_counts().reset_index().sort_values('index').rename(
        columns={'index': 'horse_id', 'horse_id': 'count'})
    horse_id_count = list(map(int, query_dataset['horse_id'].unique()))
    firstLoop = True
    print("shift horse half way shift...")
    for horse_id in tqdm(horse_id_count):
        df_tm_horse_ = df[df['horse_id'] == horse_id]
        df_tm_horse_1 = df_tm_horse_['half_way_rank'].shift(1).rename('half_way_rank_1')
        df_tm_horse_2 = df_tm_horse_['half_way_rank'].shift(2).rename('half_way_rank_2')
        df_tm_horse_3 = df_tm_horse_['half_way_rank'].shift(3).rename('half_way_rank_3')
        df_tm_horse_half_way = pd.concat([df_tm_horse_1, df_tm_horse_2, df_tm_horse_3], axis=1)
        df_tm_horse_half_way = df_tm_horse_half_way.mean(axis=1)
        df_tm_horse = pd.concat([df_tm_horse_, df_tm_horse_half_way], axis=1)
        if firstLoop:
            df_tm = df_tm_horse
            firstLoop = False
        else:
            df_tm = pd.concat([df_tm, df_tm_horse], axis=0)
    df = df_tm.drop('date', axis=1)
    df.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'horse_half_way.csv'), encoding='utf_8_sig', index=False)
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'cleaned_horse_data.csv'))
    # df = shift_df(df)
    df = half_way_rank_ave(df)
