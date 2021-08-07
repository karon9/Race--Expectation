import pandas as pd
from pathlib import Path
import os
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, date, time
import numpy as np

if __name__ == '__main__':
    url = 'https://race.netkeiba.com/race/shutuba.html?race_id=202104030511&rf=race_submenu'
    # race_id
    race_id = url.split("=")[-1]
    html = requests.get(url)
    race_list = []
    soup = BeautifulSoup(html.content, "html.parser")

    Data_list = soup.find("dl", id="RaceList_DateList")
    Data_list = Data_list.find("dd", class_="Active").find("a").get_text()
    Data_list = Data_list.split('(')[0]
    month = int(Data_list.split('月')[0])
    day = int(Data_list.split('月')[1].split('日')[0])

    race_data_01 = soup.find("div", class_="RaceData01")
    race_data_01 = race_data_01.get_text().strip("\n").split("/")
    Date = race_data_01[0]
    time = int(re.split('発走', Date)[0].split(':')[0])
    minutes = int(re.split('発走', Date)[0].split(':')[1])
    ground_type_ = race_data_01[1].split("(")[0]
    ground_type = re.split("\d+", ground_type_)[0]  # ground_type
    distance = re.split("芝|ダ|m", ground_type_)[1]  # distance
    date = datetime(2021, month, day, time, minutes)  # date

    race_data_02 = soup.find("div", class_="RaceData02")
    race_data_02 = race_data_02.get_text().strip("\n").split("\n")
    where_racecourse = race_data_02[1]  # where_racecourse
    race_rank_age = race_data_02[3].strip("サラ系")
    race_rank_rank = race_data_02[4]
    if race_rank_rank == '1勝クラス':
        race_rank_rank = '500万下'
    if race_rank_rank == '2勝クラス':
        race_rank_rank = '1000万下'
    if race_rank_rank == 'オープン':
        race_rank_rank = input('オープン戦のランクを入力してください(G1,G2,G3)\n')
    race_rank = race_rank_age + race_rank_rank  # race_rank

    horse_list = soup.find("div", class_="RaceTableArea").find_all("tr", class_="HorseList")
    horse_number_list = []
    horse_id_list = []
    rider_id_list = []
    for number in range(0, len(horse_list)):
        horse_number_list.append(number + 1)
        # horse_id
        horse_id_list.append(horse_list[number].find_all("a")[0].get("href").split("/")[-1])
        # rider_id
        rider_id_list.append(horse_list[number].find_all("a")[1].get("href").split("/")[-2])


    race_id_csv = pd.Series(np.arange(len(horse_number_list)), name="race_id")
    race_id_csv[:] = race_id
    horse_number_csv = pd.Series(horse_number_list, name="horse_number")
    rider_id_csv = pd.Series(rider_id_list, name="rider_id")
    horse_id_csv = pd.Series(horse_id_list, name="horse_id")

    horse_flow = {}
    horse_flow_list =[]
    count = 0
    race_slide_list = soup.find("div", class_="DeployRace_Slide").find("ul").find_all("dd")
    for race_slides in race_slide_list:
        race_slide = race_slides.find_all("li")
        for horse_num in race_slide:
            count += 1
            horse_name = horse_num.get_text()
            horse_num = re.split('\D', horse_name)[0]
            horse_flow[f'{count}'] = int(horse_num)

    horse_flow_sorted = sorted(horse_flow.items(), key=lambda x:x[1])
    for horse_flow in horse_flow_sorted:
        horse_flow_list.append(horse_flow[0])
    half_way_csv = pd.Series(horse_flow_list, name="half_way_rank")


    remarks_csv = pd.Series(np.arange(len(horse_number_list)), name="remarks")
    remarks_csv[:] = ''
    date_csv = pd.Series(np.arange(len(horse_number_list)), name="date")
    date_csv[:] = date
    where_racecourse_csv = pd.Series(np.arange(len(horse_number_list)), name="where_racecourse")
    where_racecourse_csv[:] = where_racecourse

    race_rank_csv = pd.Series(np.arange(len(horse_number_list)), name="race_rank")
    race_rank_csv[:] = race_rank
    baba_index_csv = pd.Series(np.arange(len(horse_number_list)), name="baba-index")
    baba_index_csv[:] = int(input('baba-indexを入力してください\n'))  # baba-index
    ground_type_csv = pd.Series(np.arange(len(horse_number_list)), name="ground_type")
    ground_type_csv[:] = ground_type
    distance_csv = pd.Series(np.arange(len(horse_number_list)), name="distance")
    distance_csv[:] = distance
    predict_csv = pd.concat(
        [race_id_csv, horse_number_csv, rider_id_csv, horse_id_csv, half_way_csv, remarks_csv, date_csv,
         where_racecourse_csv, race_rank_csv, baba_index_csv, ground_type_csv, distance_csv], axis=1)
    predict_csv.to_csv(os.path.join(Path(os.getcwd()).parent, 'predict.csv'), index=False, header=True,
                       encoding="utf_8_sig")
