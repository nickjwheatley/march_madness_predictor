import pandas as pd
import datetime as dt
from data_cleaning import web_scrape, clean_data, transform_data
import os

# Set data extraction parameters
branch='men'
lookback_years = 4 # number of years of data to consider including current year
this_year = dt.datetime.now().year

# Get season dates and years of data to extract
season_dates = pd.read_csv('data/season_dates.csv',
                           index_col='season_end_year',
                           infer_datetime_format=True,
                           parse_dates=['season_start_date','tournament_start_date','season_end_date'])

year_ix = season_dates.index.get_loc(this_year) + 1
extraction_years = season_dates.index[year_ix-lookback_years:year_ix].tolist()

if this_year not in extraction_years:
    raise Exception(f'Season dates for season ending in {this_year} are not included in ../data/season_dates.csv')

# Forces model to extract data, cache it locally, and transform it (add calculated metrics)
force_extract = False
force_transform = False

# EXTRACT AND CLEAN DATA
raw_datas = web_scrape.extract_all_data(extraction_years, season_dates, this_year, branch, force_extract)
cleaned_datas = clean_data.clean_data(raw_datas, this_year, branch)
data = transform_data.transform_data(cleaned_datas, this_year, lookback_years, branch, force_transform)

# Exclude qualitative columns for model training
drop_columns = ['team','opponent','team_score','opponent_score','game_round','season_type','date','season_year',
                'team_rank','opponent_rank','seed','seed_opp','home_game','g','w','l','g_opp','w_opp','l_opp',
                'plus_minus','conf','conf_opp']
df_features = data.drop(drop_columns, axis=1)

features_filepath = 'data/model_features.csv'
if (not os.path.exists(features_filepath)) | force_transform:
    df_features.to_csv(features_filepath)

import configparser
config = configparser.ConfigParser()
config.read('secrets.ini')
print(config['gpt4']['token'])