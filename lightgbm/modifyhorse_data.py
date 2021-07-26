import pandas as pd
import os




if __name__ == '__main__':
    race_df = pd.read_csv(os.path.join('csv', 'cleaned_horse_data.csv'))
    horse_df = pd.read_csv(os.path.join('csv', 'cleaned_race_data.csv'))
