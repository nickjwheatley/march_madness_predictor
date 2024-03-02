#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import numpy as np
import os
import datetime as dt


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


def transform_data(datas,this_year,lookback,branch='men', force=False):
    data_years = list(range(this_year-lookback+1,this_year+1))
    data_years_str = f'{str(data_years[0])[-2:]}_to_{str(data_years[-1])[-2:]}'
    tranformed_filepath = f'data/transformed_data_{branch}_{data_years_str}.csv'

    if os.path.exists(tranformed_filepath) & (not force):  # & (not force):
        print(f'{this_year} data already transformed')
        data = pd.read_csv(tranformed_filepath)
    else:
        sos_scaled = False

        # Extract all cleaned data from data_extraction notebook
        if branch == 'men':
            ncaa,sportsref,bart = datas
        else:
            ncaa,sportsref = datas

        bart.seed = pd.to_numeric(bart.seed)

        # Create a normalized index of certain metrics for better comparison
        # for metric in ['orb','pf','stl','blk','tov','sos','srs']:
        #     sportsref = min_max_normalize(metric,sportsref,extraction_years)

        # Convert certain metrics to their per-game ratio
        per_game_metrics = ['fg', 'fga', '3p', '3pa', 'ft', 'fta', 'orb', 'trb', 'ast', 'stl', 'blk', 'pf', 'tov']
        for metric in per_game_metrics:
            sportsref[metric] = sportsref[metric] / sportsref['g']

        # Create physicality score
        physicality_metrics = ['orb','stl','blk','tov']
        physicality_weight = 1/len(physicality_metrics)

        if sos_scaled:
            sportsref['physicality_score'] = sportsref[physicality_metrics].apply(
                lambda x:sum(x*physicality_weight),axis=1) * (sportsref['sos'] / 100)
        else:
            sportsref['physicality_score'] = sportsref[physicality_metrics].apply(
                lambda x:sum(x*physicality_weight),axis=1)


        # Create Offensive Efficiency
        if sos_scaled:
            sportsref['oe'] = \
                sportsref['tm.'] / ((sportsref['mp'] / 40 * sportsref['pace']) / 100) \
                * (sportsref['sos'] / 100)
        else:
            sportsref['oe'] = \
                sportsref['tm.'] / ((sportsref['mp'] / 40 * sportsref['pace']) / 100)

        # Create Defensive effienciency
        if sos_scaled:
            sportsref['de'] = \
                sportsref['opp.'] / ((sportsref['mp'] / 40 * sportsref['pace_opp']) / 100) \
                * (sportsref['sos'] / 100)
        else:
            sportsref['de'] = \
                sportsref['opp.'] / ((sportsref['mp'] / 40 * sportsref['pace_opp']) / 100)

        # Create Total Efficiency
        if sos_scaled:
            sportsref['te'] = \
                sportsref['oe'] / sportsref['de'] * (sportsref['sos'] / 100)
        else:
            sportsref['te'] = \
                sportsref['oe'] / sportsref['de']

        # Add predicted possessions
        sportsref['poss'] = sportsref['fg'] - sportsref['orb'] + sportsref['tov'] + 0.475 * sportsref['fta']

        # Add Assists per Possesion
        sportsref['ast_per_poss'] = sportsref['ast'] / sportsref['poss']

        # Add Assists per FGM
        sportsref['ast_per_fg'] = sportsref['ast'] / sportsref['fg']

        # Add Turnovers per Possession
        sportsref['tov_per_poss'] = sportsref['tov'] / sportsref['poss']

        # Add Assists to Turnovers ratio
        sportsref['ast_to_tov'] = sportsref['ast'] / sportsref['tov']

        # # Possessions per Game
        # sportsref['poss_per_game'] = sportsref['poss'] / sportsref['g']

        # Add Defensive Rebounds
        sportsref['drb'] = sportsref['trb'] - sportsref['orb']

        # Season Game Win Rate
        sportsref['game_win_rate'] = sportsref['w'] / sportsref['g']

        # Add GPT Scores
        gpt_score_filepath = 'data/team_gpt_score.csv'
        gpt_score_add = []
        if os.path.exists(gpt_score_filepath):
            gpt_scores = pd.read_csv(gpt_score_filepath)
            sportsref = sportsref.merge(gpt_scores, on='team', how='left')
            gpt_score_add = ['gpt_sent_score_avg']

        # Combine sport-reference and ncaa dataframes
        desired_cols = ['team','season_year','g','w','l','de','oe','te','pace','physicality_score','sos','srs','fg',
                        'fga','fg%','3p','3pa','3p%','ft','fta','ft%','orb','drb','trb','ast','stl','blk','pf','tov',
                        'tov%', 'poss','ast_per_poss','ast_per_fg','tov_per_poss','ast_to_tov','poss_per_game',
                        'game_win_rate'] + gpt_score_add
        opponent_rename = {col:col+'_opp' for col in desired_cols if col not in ['team','season_year']}
        opponent_rename['team'] = 'opponent'

        opponent_stats = sportsref[desired_cols].rename(columns=opponent_rename)

        data = ncaa.merge(sportsref[desired_cols],on=['team','season_year'],how='left')
        data = data.merge(opponent_stats,on=['opponent','season_year'],how='left')
        data.dropna(subset=['te','te_opp'],inplace=True)

        if branch == 'men':
            # Merge Bart data for tournament seeds
            data = data.merge(bart[['team','season_year','seed','conf']],on=['team','season_year'],how='left')
            data = data.merge(bart[['team','season_year','seed','conf']].rename(columns={
                'team':'opponent','seed':'seed_opp','conf':'conf_opp'}),on=['opponent','season_year'],how='left')

            # # Add team conference
            # data = data.merge(bart[['team', 'season_year', 'seed', 'conf']], on=['team', 'season_year'], how='left')
            # data = data.merge(bart[['team', 'season_year', 'seed', 'conf']].rename(columns={
            #     'team': 'opponent', 'seed': 'seed_opp', 'conf': 'conf_opp'}), on=['opponent', 'season_year'],
            #     how='left')

            # Solve for missing conferences
            most_recent_conferences = bart[['team','conf']].drop_duplicates(
                subset='team',keep='last').rename(columns={'conf': 'most_recent_conf'})
            data = data.merge(most_recent_conferences, on='team', how='left')
            data = data.merge(most_recent_conferences.rename(
                columns={'team':'opponent','most_recent_conf':'most_recent_conf_opp'}), on='opponent', how='left')

            data.loc[data.conf.isna(),'conf'] = data.loc[data.conf.isna(),'most_recent_conf']
            data.loc[
                data.conf_opp.isna(), 'conf_opp'] = data.loc[data.conf_opp.isna(), 'most_recent_conf_opp']

            data.drop(['most_recent_conf','most_recent_conf_opp'], axis=1, inplace=True)
            data.loc[data.team == 'Saint Francis (NY)', 'conf'] = 'NEC'
            data.loc[data.opponent == 'Saint Francis (NY)', 'conf_opp'] = 'NEC'

            # Rank conferences by the number of tournament games won in prior years data
            ranked_conferences = bart.loc[
                bart.season_year < dt.datetime.now().year,
                ['conf','games_won']].groupby('conf').sum().sort_values('games_won', ascending=False).reset_index()

            # Add missing conferences
            missing_conferences = [
                (conf,0) for conf in bart.conf.unique() if conf not in ranked_conferences.conf.tolist()]
            df_missing_conferences = pd.DataFrame(missing_conferences, columns = ['conf','games_won'])

            ranked_conferences = pd.concat([ranked_conferences,df_missing_conferences])
            ranked_conferences['conf_rank'] = ranked_conferences['games_won'].rank(method='min', ascending=False)

            conf_ints = {conf:i for conf,i in ranked_conferences[['conf','conf_rank']].values}
            data['conf_rank'] = data.conf.replace(conf_ints)
            data['conf_rank_opp'] = data.conf_opp.replace(conf_ints)


        # Create Luck Factor Metric
        data['plus_minus'] = data['team_score'] - data['opponent_score']
        data['close_game'] = 0
        data.loc[abs(data.plus_minus) <= 4,'close_game'] = 1

        luck_data = data.loc[data.close_game == 1,['team','season_year','won']].copy()

        luck = (luck_data.groupby(['team','season_year',]).sum() \
                / luck_data.groupby(['team','season_year']).count()).reset_index()


        data = data.merge(luck.rename(columns={'won':'luck'}),on=['team','season_year'],how='left')
        data = data.merge(luck.rename(columns={'won':'luck_opp','team':'opponent'}),on=['opponent','season_year'],how='left')
        data.drop(['close_game'],axis=1,inplace=True)

        # data.luck.fillna(0,inplace=True)
        # data.luck_opp.fillna(0, inplace=True)


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

        # fillna with 0
        # data.choke_rate.fillna(0, inplace=True)
        # data.choke_rate_opp.fillna(0, inplace=True)

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

        # fillna with 0
        # data.upset_rate.fillna(0, inplace=True)
        # data.upset_rate_opp.fillna(0, inplace=True)

        # Calculate average plus_minus
        data.sort_values(['team', 'date'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data['3mean_plus_minus'] = data[
            ['team','season_year','plus_minus']].groupby(['team','season_year']).shift(1).rolling(3).mean()

        data['3mean_plus_minus'].fillna(method='bfill', inplace=True)

        plus_minus = data[['team', 'date', '3mean_plus_minus']].copy()

        # add opponent win_streaks
        data = data.merge(
            plus_minus.rename(columns={'team': 'opponent', '3mean_plus_minus': '3mean_plus_minus_opp'}),
            on=['opponent', 'date'],
            how='left'
        )

        # Create Win-Streak metric scaled by strength of schedule
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
        # data.win_streak = data.win_streak * (data.sos_norm/100)
        # data.win_streak_opp = data.win_streak_opp * (data.sos_norm_opp/100)

        # Calculate matchup win rate
        print('Calculating matchup win rate')
        matchups_df = pd.DataFrame()

        matchups = data[['team','opponent','season_year']].drop_duplicates().values.tolist()
        match_wl = []
        for match in matchups:
            wins = data.loc[
                (data.team == match[0]) &
                (data.opponent == match[1]) &
                (data.season_year <= match[2]),'won'
            ].sum()
            games = data.loc[
                (data.team == match[0]) &
                (data.opponent == match[1]) &
                (data.season_year <= match[2]),'won'
            ].count()

            # Remove model's ability to predict using 1-off events
            wins -= 1
            games -= 1

            wins = max(wins,0)
            games = max(games,0)
            win_rate = np.nan if games == 0 else wins/games
            match_wl.append(match+[win_rate])

        matchups_df = pd.DataFrame(match_wl,columns=['team','opponent','season_year','matchup_win_rate'])
        data = data.merge(matchups_df,on=['team','opponent','season_year'],how='left')

        # data.matchup_win_rate.fillna(0,inplace=True)
        data = data.drop_duplicates()
        data.to_csv(tranformed_filepath,index=False)
    return data


# In[345]:


# Avg Win Surplus multiplied by SOS (combined)
# plus_minus = data[
#     ['team','season_year','plus_minus']].groupby(
#     ['team','season_year']).mean().reset_index().rename(columns={'plus_minus':'avg_plus_minus'})

