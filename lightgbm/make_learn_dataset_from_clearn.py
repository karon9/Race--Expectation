import pandas as pd
import os
from drop_dataset import drop_race_data, drop_horse_data
from pathlib import Path

horse_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_cleaned_horse_data.csv'))
race_data = pd.read_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_cleaned_race_data.csv'))
horse_data = horse_data.sort_values('race_id', ascending=True)
race_data = race_data.sort_values('race_id', ascending=True)
horse_data = drop_horse_data(horse_data)
race_data = drop_race_data(race_data)
query_dataset = pd.DataFrame(horse_data.groupby('race_id')['horse_number'].count()).reset_index(drop=True)
query_dataset.to_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_query_data.csv'), index=False)
data = pd.merge(horse_data, race_data, on='race_id')
data.to_csv(os.path.join(Path(os.getcwd()).parent, 'test_csv', 'test_data.csv'), encoding='utf_8_sig', index=False)
