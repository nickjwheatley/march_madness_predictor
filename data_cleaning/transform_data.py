#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import datetime as dt
import os

from data_cleaning import clean_data


# In[ ]:


def min_max_normalize(metric,data,years):
    """
    Min-Max Normalize a given metric in a specific DataFrame
    :param metric: string name indicating column to be normalized
    :param data: pandas DataFrame containing said metric
    :param years: list of all years in the extraction
    :return pandas DataFrame containing the normalized metric
    """
    dfs = []
    for year in years:
        tmp_df = data.loc[data.season_year == year].copy()
        tmp_df[f'{metric}_norm'] = \
                (tmp_df[metric] - min(tmp_df[metric])) / \
                (max(tmp_df[metric]) - min(tmp_df[metric])) * 100
        dfs.append(tmp_df)
    data1 = pd.concat(dfs)
    return data1


# In[353]:


def calculate_win_streak(team,data):
    streaks = []
    tmp_data = data.loc[data.team == team].sort_values('date')
    wins = tmp_data.won.tolist()
    dates = tmp_data.date.tolist()
    
    streak = 0
    for w in wins:
        if w == 1:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    return streaks


# In[354]:


def transform_data(datas, extraction_years,this_year,branch='men'):
    tranformed_filepath = f'data/transformed_data_{branch}{str(this_year)[-2:]}.csv'
    sos_scaled = False

    # Extract all cleaned data from data_extraction notebook
    if branch == 'men':
        ncaa,sportsref,bart = datas
    else:
        ncaa,sportsref = datas

    # Create a normalized index of certain metrics for better comparison
    for metric in ['orb','pf','stl','blk','tov','sos','srs']:
        sportsref = min_max_normalize(metric,sportsref,extraction_years)

    # Create physicality score
    physicality_metrics = ['orb_norm','pf_norm','stl_norm','blk_norm','tov_norm']
    physicality_weight = 1/len(physicality_metrics)

    if sos_scaled:
        sportsref['physicality_score'] = sportsref[physicality_metrics].apply(
            lambda x:sum(x*physicality_weight),axis=1) * (sportsref['sos_norm'] / 100)
    else:
        sportsref['physicality_score'] = sportsref[physicality_metrics].apply(
            lambda x:sum(x*physicality_weight),axis=1)


    # Create Offensive Efficiency
    if sos_scaled:
        sportsref['oe'] = \
            sportsref['tm.'] / ((sportsref['mp'] / 40 * sportsref['pace']) / 100) \
            * (sportsref['sos_norm'] / 100)
    else:
        sportsref['oe'] = \
            sportsref['tm.'] / ((sportsref['mp'] / 40 * sportsref['pace']) / 100)

    # Create Defensive effienciency
    if sos_scaled:
        sportsref['de'] = \
            sportsref['opp.'] / ((sportsref['mp'] / 40 * sportsref['pace_opp']) / 100) \
            * (sportsref['sos_norm'] / 100)
    else:
        sportsref['de'] = \
            sportsref['opp.'] / ((sportsref['mp'] / 40 * sportsref['pace_opp']) / 100)

    # Create Total Efficiency
    if sos_scaled:
        sportsref['te'] = \
            sportsref['oe'] / sportsref['de'] * (sportsref['sos_norm'] / 100)
    else:
        sportsref['te'] = \
            sportsref['oe'] / sportsref['de']


    # Combine sport-reference and ncaa dataframes
    desired_cols = ['team','season_year','de','oe','te','pace','physicality_score','sos_norm','srs_norm']
    opponent_rename = {col:col+'_opp' for col in desired_cols if col not in ['team','season_year']}
    opponent_rename['team'] = 'opponent'

    opponent_stats = sportsref[desired_cols].rename(columns=opponent_rename)

    data = ncaa.merge(sportsref[desired_cols],on=['team','season_year'],how='left')
    data = data.merge(opponent_stats,on=['opponent','season_year'],how='left')
    data.dropna(subset=['te','te_opp'],inplace=True)

    if branch == 'men':
        # Merge Bart data for tournament seeds
        data = data.merge(bart[['team','season_year','seed']],on=['team','season_year'],how='left')
        data = data.merge(bart[['team','season_year','seed']].rename(columns={
            'team':'opponent','seed':'seed_opp'}),on=['opponent','season_year'],how='left')


    # Create Luck Factor Metric
    data['score_diff'] = data['team_score'] - data['opponent_score']
    data['close_game'] = 0
    data.loc[abs(data.score_diff) <= 4,'close_game'] = 1

    luck_data = data.loc[data.close_game == 1,['team','season_year','won']].copy()

    luck = (luck_data.groupby(['team','season_year',]).sum() \
            / luck_data.groupby(['team','season_year']).count()).reset_index()


    data = data.merge(luck.rename(columns={'won':'luck'}),on=['team','season_year'],how='left')
    data = data.merge(luck.rename(columns={'won':'luck_opp','team':'opponent'}),on=['opponent','season_year'],how='left')
    data.drop(['close_game'],axis=1,inplace=True)


    # Create choke metric
    data['chokable'] = 0

    if branch == 'men':
        data.loc[
            ((data.season_type == 'regular_season') & (data.underdog_opp == 1)) |
            ((data.season_type == 'ncaa_tournament') &
             ((data.seed - data.seed_opp) > 8)),'chokable'] = 1
    else:
        data.loc[data.underdog_opp == 1,'chokable'] = 1

    choke_data = data.loc[data.chokable == 1,['team','season_year','won']].copy()

    choke = (1 - choke_data.groupby(['team','season_year']).sum() \
            / choke_data.groupby(['team','season_year']).count()).reset_index()

    # merge choke data for team and opponent
    data = data.merge(choke.rename(columns={'won':'choke_rate'}),on=['team','season_year'],how='left')
    data = data.merge(choke.rename(columns={'team':'opponent','won':'choke_rate_opp'}),on=['opponent','season_year'],how='left')

    data.drop('chokable',axis=1,inplace=True)


    # Create Upset Metric
    data['upsettable'] = 0
    if branch == 'men':
        data.loc[
            ((data.season_type == 'regular_season') & (data.underdog == 1)) |
            ((data.season_type == 'ncaa_tournament') &
             ((data.seed - data.seed_opp) <= 8)),'upsettable'] = 1
    else:
        data.loc[data.underdog == 1,'upsettable'] = 1

    upset_data = data.loc[data.upsettable == 1,['team','season_year','won']].copy()

    upset = (upset_data.groupby(['team','season_year']).sum() \
            / upset_data.groupby(['team','season_year']).count()).reset_index()

    # Merge choke data for team and opponent
    data = data.merge(upset.rename(columns={'won':'upset_rate'}),on=['team','season_year'],how='left')
    data = data.merge(upset.rename(columns={'team':'opponent','won':'upset_rate_opp'}),on=['opponent','season_year'],how='left')

    data.drop('upsettable',axis=1,inplace=True)


    # Create Win-Streak metric scaled by strength of schedule
    data.sort_values(['team','date'],inplace=True)
    data.reset_index(drop=True,inplace=True)
    data['win_streak'] = 0
    for team in data.team.unique():
        data.loc[data.team==team,'win_streak'] = calculate_win_streak(team,data)


    # Shift win-streak down 1 game to show previous games win-streak
#         data.win_streak = data.win_streak.apply(lambda x: x-1 if x != 0 else 0)
    win_streaks = data[['team','date','win_streak']].copy()
    win_streaks['win_streak'] = win_streaks[['team','win_streak']].groupby(['team']).shift(1).fillna(0).astype(int)

    data.drop('win_streak',axis=1,inplace=True)
    # add team win_streaks
    data = data.merge(win_streaks,on=['team','date'],how='left')

    # add opponent win_streaks
    data = data.merge(
        win_streaks.rename(columns={'team':'opponent','win_streak':'win_streak_opp'}),
        on=['opponent','date'],
        how='left'
    )

    # # Normalize win-streak
    data.win_streak = data.win_streak * (data.sos_norm/100)
    data.win_streak_opp = data.win_streak_opp * (data.sos_norm_opp/100)

    # Calculate matchup win rate
    print('Calculating matchup win rate')
    matchups_df = pd.DataFrame()

    matchups = data[['team','opponent']].drop_duplicates().values.tolist()
    match_wl = []
    for match in matchups:
        wins = data.loc[
            (data.team == match[0]) &
            (data.opponent == match[1]),'won'
        ].sum()
        games = data.loc[
            (data.team == match[0]) &
            (data.opponent == match[1]),'won'
        ].count()

        # Remove model's ability to predict using 1-off events
        wins -= 1
        games -= 1

        wins = max(wins,0)
        games = max(games,0)
        match_wl.append(match+[wins/games])

    matchups_df = pd.DataFrame(match_wl,columns=['team','opponent','matchup_win_rate'])
    data = data.merge(matchups_df,on=['team','opponent'],how='left')

    data.to_csv(tranformed_filepath,index=False)
        
    return data


# In[345]:


# Avg Win Surplus multiplied by SOS (combined)
# score_diff = data[
#     ['team','season_year','score_diff']].groupby(
#     ['team','season_year']).mean().reset_index().rename(columns={'score_diff':'avg_score_diff'})

