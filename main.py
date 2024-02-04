import pandas as pd
import datetime as dt
from data_cleaning import web_scrape, clean_data

# Set data extraction parameters
branch='men'
lookback_years = 2 # number of years of data to consider including current year
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

# Forces model to re-extract data and cache it locally
force_extract = False

# EXTRACT AND CLEAN DATA
raw_datas = web_scrape.extract_all_data(extraction_years, season_dates, this_year, branch, force_extract)
cleaned_datas = clean_data.clean_data(raw_datas, this_year, branch)
