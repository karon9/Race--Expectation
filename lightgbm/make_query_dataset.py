import pandas as pd
import os


def test(horse_data: pd.DataFrame, race_data: pd.DataFrame):
    return None


if __name__ == '__main__':
    horse_data = pd.read_csv(os.path.join(os.getcwd(), 'test_csv', 'test_horse_data.csv'))
    query_dataset = pd.DataFrame(horse_data.groupby('race_id')['horse_number'].count()).reset_index(drop=True)
    query_dataset.to_csv(os.path.join(os.getcwd(), 'test_csv', 'test_query_data.csv'), index=False)
