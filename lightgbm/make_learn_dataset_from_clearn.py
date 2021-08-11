import pandas as pd
import os
from drop_dataset import drop_race_data, drop_horse_data
from pathlib import Path
from modify_horse import shift_df


def main():
    horse_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'cleaned_horse_data.csv'))
    race_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'cleaned_race_data.csv'))
    horse_data = shift_df(horse_data)
    horse_data = horse_data.sort_values(['race_id', 'rank'], ascending=[True, True])
    race_data = race_data.sort_values(['date', 'race_id'], ascending=[True, True])
    horse_data = drop_horse_data(horse_data)
    race_data = drop_race_data(race_data)

    data = pd.merge(horse_data, race_data, on='race_id').sort_values(['date', 'horse_number'], ascending=[True, True])
    query_dataset = pd.DataFrame(data.groupby('date')['horse_number'].count()).reset_index(drop=True)

    query_dataset.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'learn_query_data.csv'), index=False)

    data.to_csv(os.path.join(Path(os.getcwd()).parent, 'csv', 'learn_data.csv'), encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    main()
