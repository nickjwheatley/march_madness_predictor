#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import bs4
import urllib
import datetime as dt
import os
import numpy as np
import requests


# ## Extract data from 3 main sites:
# 
# ### ncaa.com
# - Rank
# - Individual game data
# 
# ### sports-reference.com
# - Offensive Rebounds
# - Personal Fouls
# - Steals
# - Blocks
# - Strength of Schedule
# 
# ### barttorvik.com (just for mens)
# - ncaa teams and seeds

# In[2]:


def extract_text(item):
    return item.text


# In[3]:


def get_ranks(item):
    if item != '':
        return int(item)
    else:
        return np.nan


# In[4]:


def get_scores(item):
    if item == '':
        return np.nan
    else:
        return int(item)


# In[5]:


def get_game_round(item):
    if len(item) == 0:
        return ''
    else:
        return item[0].text.replace('®','') # Remove copyright symbol


# In[6]:


def extract_ncaa_data(year,season_start_date,tournament_start_date,season_end_date,branch='men'):
    days = (season_end_date - season_start_date).days
    dfs = []
    for d in range(days+1):
        date = (season_start_date + dt.timedelta(d)).strftime('%Y/%m/%d')
        print(date)

        url = f'https://www.ncaa.com/scoreboard/basketball-{branch}/d1/{date}/all-conf'
        url_contents = urllib.request.urlopen(url).read()
        soup = bs4.BeautifulSoup(url_contents, "html")
        div = soup.find("div", {"id": "scoreboardGames"})
    
        teams = div.find_all('span',{'class':'gamePod-game-team-name'})
        ranks = div.find_all('span',{'class':'gamePod-game-team-rank'})
        scores = div.find_all('span',{'class':'gamePod-game-team-score'})
        descriptions = div.find_all('div',{'class':'gamePod-description'})
        game_rounds = [d.find_all('span',{'class':'game-round'}) for d in descriptions] # Only during tournament
        
        teams = list(map(extract_text,teams))
        ranks = list(map(get_ranks,list(map(extract_text,ranks))))
        scores = list(map(get_scores,list(map(extract_text,scores))))
        game_round = list(map(get_game_round,game_rounds))
        
        home_teams = teams[1::2]
        away_teams = teams[::2]
        home_scores = scores[1::2]
        away_scores = scores[::2]
        home_ranks = ranks[1::2]
        away_ranks = ranks[::2]

        data = {}
        vals = [
            home_teams, away_teams,
            home_scores, away_scores,
            home_ranks, away_ranks, 
            game_round
        ]
        
        i = 0
        for c in ['home_team','away_team','home_score','away_score','home_rank','away_rank','game_round']:
            data[c] = vals[i]
            i += 1
            
        if season_start_date + dt.timedelta(d) >= tournament_start_date:
            data['season_type'] = 'ncaa_tournament'
        else:
            data['season_type'] = 'regular_season'

        df_temp = pd.DataFrame(data)
        df_temp['date'] = date
        df_temp['season_year'] = year
        dfs.append(df_temp)

    df = pd.concat(dfs)
    df.loc[(df.season_type == 'ncaa_tournament') & (df.game_round == ''),'season_type'] = 'regular_season'
    return df


# In[2]:


def get_games_won(final_round):
    games_won = {
        'R68':0,'R64':0,'R32':1,'Sweet Sixteen':2,'Elite Eight':3,
        'Final Four':4,'Finals':5,'CHAMPS':6,'✅':'TBD','❌':'TBD'
    }
    return games_won[final_round]


# In[3]:


def extract_barttorvik_data(year):
    res = requests.get('https://barttorvik.com/trank.php?year={}&sort=&top=0&conlimit=All&venue=All&type=All#'.format(year))
    html = pd.read_html(res.content)
    df = pd.DataFrame(html[0])
    df.columns = df.columns.droplevel(level=0)
    df = df.loc[df['Rk'] != 'Rk'][['Team','Conf','AdjOE','AdjDE']].rename(
        columns={'AdjOE':'Offensive Efficiency','AdjDE':'Defensive Efficiency'})

    for col in ['Offensive Efficiency','Defensive Efficiency']:
        df[col] = df[col].apply(lambda x: x.split()[0])
        df[col] = pd.to_numeric(df[col])
    df['Total Efficiency'] = df['Offensive Efficiency'] / df['Defensive Efficiency']

    df.insert(1,'Seed',df['Team'].str.extract('([0-9]+) seed'))
    df.insert(2,'Final Round',df['Team'].str.extract(', (.*)'))

    current_season_flag = False
    if (year == dt.datetime.now().year) & (dt.datetime.now().month < 3):
        current_season_flag = True
        # df.Seed.fillna(np.nan, inplace=True)
        df['Final Round'].fillna('TBD', inplace=True)

    # df['Team'] = df['Team'].str.extract('(.*) [1-9]')
    df['Team'] = df['Team'].str.extract('^([^\d(]+)') # Clean team names
    df['Team'] = df['Team'].apply(lambda x: x.strip())
    df.insert(0,'Season Year',year)

    if not current_season_flag:
        df.dropna(inplace=True)

    #
    if current_season_flag:
        df.insert(4,'Games Won','TBD')
    else:
        df.insert(4,'Games Won',list(map(get_games_won,df['Final Round'])))
    
    # Make column names lower_case
    lower_cols = {x:'_'.join(x.lower().split(' ')) for x in df.columns} # change columns to lowercase
    df = df.rename(columns=lower_cols)
    return df


# In[28]:


def extract_sportsref_data(year,this_year,team_opp='school',basic_adv='basic',branch='men'):
    # Extract data from site sports-reference.com
    if basic_adv == 'basic':
        url = f'https://www.sports-reference.com/cbb/seasons/{branch}/{year}-{team_opp}-stats.html'
    else:
        url = f'https://www.sports-reference.com/cbb/seasons/{branch}/{year}-advanced-{team_opp}-stats.html'
    res = requests.get(url)
    df = pd.read_html(res.content)[0]

    # Drop undesired sections
    drop_sections = ['Conf.','Home','Away']
    df.drop(drop_sections,axis=1,inplace=True)

    # Remove level 0 column multi-index
    df.columns = df.columns.droplevel(level=0)

    # Replace 'Rk' elements in the 'Rk' column with nans to be dropped later
    df['Rk'] = df['Rk'].replace('Rk',np.nan)

    # Drop nans in Rk column to remove section dividers
    df.dropna(subset=['Rk'],inplace=True)
    
    # Rename school to team to match other datasets
    df = df.rename(columns={'School':'team'})
    
    # Specify desired column ranges depending on site retrieved
    if (team_opp == 'school') & (basic_adv == 'basic'):
        desired_columns = df.columns[1:8].tolist() + df.columns[12:14].tolist()+ df.columns[15:].tolist()
    elif (team_opp == 'opponent') & (basic_adv == 'basic'):
        desired_columns = ['team','ORB','TRB','STL','TOV']
    elif (team_opp == 'school') & (basic_adv == 'advanced'):
        desired_columns = ['team','Pace','TOV%']
    else:
        desired_columns = ['team','Pace','TOV%']
    
    # Narrow columns
    lower_cols = {x:x.lower() for x in df.columns} # change columns to lowercase
    if team_opp == 'opponent':
        for col in lower_cols.keys():
            if col != 'team':
                lower_cols[col] = lower_cols[col]+'_opp'
    df = df[desired_columns].rename(columns=lower_cols)
    
    # No Barttorvik data exists for women's league - must grab data from sportsref
    if (year == this_year) & (branch == 'women'):
        ncaa_qualifiers_w = df.copy()
        ncaa_qualifiers_w['ncaa_qualifier'] = ncaa_qualifiers_w.team.apply(
            lambda x:1 if 'NCAA' in x else 0)
        ncaa_qualifiers_w = ncaa_qualifiers_w.loc[
            ncaa_qualifiers_w.ncaa_qualifier==1,['team','ncaa_qualifier']]
        
    # Clean team names
    # Replace bad text
    replacers_empty = [r'\xa0NCAA']
    replacers_space = ['-']

    for r in replacers_empty:
        df.team = df.team.str.replace(r,'',case=True, regex=True)
        if (year == this_year) & (branch == 'women'):
            ncaa_qualifiers_w.team = ncaa_qualifiers_w.team.str.replace(r,'')

    for r in replacers_space:
        df.team = df.team.str.replace(r,' ',case=True, regex=True)
        if (year == this_year) & (branch == 'women'):
            ncaa_qualifiers_w.team = ncaa_qualifiers_w.team.str.replace(r,' ')
            
    try:
        if (year == this_year) & (branch == 'women') & (len(ncaa_qualifiers_w) == 0):
            print(f'Women NCAA Qualifiers not yet posted to Sports-Reference.com for {year}')
        else:
            ncaa_qualifiers_w[['team']].to_csv(
                f'data/ncaa_qualifiers_w{str(year)[-2:]}.csv',index=False)
    except:
        pass
            
    return df.set_index('team')


# In[51]:


def merge_sportsdef_data(year,this_year,branch):
    dfs = []
    for org_type in ['school','opponent']:
        for metric_type in ['basic','advanced']:
            dfs.append(extract_sportsref_data(year,this_year,org_type,metric_type,branch))

    df = pd.concat(dfs,axis=1).reset_index()
    df['season_year'] = year
    return df


# In[74]:


# Get current year to determine extraction
def extract_all_data(extraction_years,season_dates,this_year=2023,branch='men', force=False):
    if this_year not in extraction_years:
        raise Exception(f'Season dates for season ending in {this_year} are not included in data/season_dates.csv')

    ncaas = []
    sportsrefs = []
    
    if branch == 'men':
        barts = []

    for year in extraction_years:
        print(year)
        # Get NCAA Data
        ncaa_filepath = f'data/ncaa{str(year)[-2:]}_{branch}.csv'
        if os.path.exists(ncaa_filepath):# & (not force):
            print(f'ncaa{year} already logged')
            ncaa = pd.read_csv(ncaa_filepath)
        else:
            season_start_date = season_dates.loc[year].season_start_date
            tournament_start_date = season_dates.loc[year].tournament_start_date
            if year == this_year:
                season_end_date = dt.datetime.today()
            else:
                season_end_date = season_dates.loc[year].season_end_date
            ncaa = extract_ncaa_data(
                year,season_start_date,tournament_start_date,season_end_date,branch)
            ncaa.to_csv(ncaa_filepath,index=False)
        ncaas.append(ncaa)

        if branch == 'men':
            # Get Barttorvik data
            bart_filepath = f'data/bart{str(year)[-2:]}.csv'
            if os.path.exists(bart_filepath) & (not force):
                print(f'bart{year} already logged')
                bart = pd.read_csv(bart_filepath)
            else:
                bart = extract_barttorvik_data(year)
                if len(bart) == 0:
                    print(f'NCAA Tournament teams for {year} not yet announced in barttorvik.com')
                else:
                    bart.to_csv(bart_filepath, index=False)
            barts.append(bart)

        # Get Sports-Reference data
        sportsref_filepath = f'data/sportsref{str(year)[-2:]}_{branch}.csv'
        if os.path.exists(sportsref_filepath) & (year != this_year) & (not force): # Continue logging the current season
            print(f'sportsref{year} already logged')
            sportsref = pd.read_csv(sportsref_filepath)
        else:
            sportsref = merge_sportsdef_data(year,this_year,branch)
            sportsref.to_csv(sportsref_filepath,index=False)
        sportsrefs.append(sportsref)

    ncaa = pd.concat(ncaas)
    sportsref = pd.concat(sportsrefs)
    
    if branch == 'men':
        bart = pd.concat(barts)
        return ncaa,sportsref,bart
    else:
        return ncaa,sportsref

