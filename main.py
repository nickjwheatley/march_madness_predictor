import pandas as pd
import numpy as np
import datetime as dt
from data_cleaning import web_scrape, clean_data, transform_data
from model.supervised.supervised_models import build_best_model
import utils.tournament_tools as tourney
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# Set data extraction parameters
branch='men'
lookback_years = 10 # number of years of data to consider including current year
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

# EXTRACT, CLEAN AND TRANSFORM DATA
raw_datas = web_scrape.extract_all_data(extraction_years, season_dates, this_year, branch, force_extract)
cleaned_datas = clean_data.clean_data(raw_datas, this_year, branch)
data = transform_data.transform_data(cleaned_datas, this_year, lookback_years, branch, force_transform)

tournament_only = True # Mark True to only train model on NCAA tournament data, else include regular season
if tournament_only:
    model_data = data.loc[data.season_type == 'ncaa_tournament']
else:
    model_data = data.copy()

# Exclude qualitative columns for model training
drop_columns = ['team','opponent','team_score','opponent_score','game_round','season_type','date','season_year',
                'team_rank','opponent_rank','seed','seed_opp','home_game','g','w','l','g_opp','w_opp','l_opp',
                'plus_minus','conf','conf_opp']
df_features = model_data.drop(drop_columns, axis=1)

features_filepath = 'data/model_features.csv'
if (not os.path.exists(features_filepath)) | force_transform | tournament_only:
    df_features.to_csv(features_filepath, index=False)

# RUN MONTE CARLO SIMULATION
num_simulations = 1000
print(f'Running Monte Carlo with {num_simulations} simulations')
for i in range(num_simulations):
    # Create best model with no random_state (for monte carlo)
    model = build_best_model(df_features)

    # RUN THE TOURNAMENT FOR A SINGLE SIMULATION
    first_four_complete = False
    first_four_losers = None
    tournament_teams_filepath = f'data/tournament_regions_{branch}{str(this_year)[-2:]}.csv'
    if not os.path.exists(tournament_teams_filepath):
        print('Missing file: ',tournament_teams_filepath)
        raise('Must add MM tournament teams by region to the data folder')
    else:
        # Get teams in their bracket regions
        teams = pd.read_csv(tournament_teams_filepath)

        # Get all non-calculated and non-qualitative columns from the last match per team
        non_calc_columns = [col for col in data.columns.tolist() if ('_opp' not in col) &\
                            (col not in drop_columns + ['underdog','fan_favorite','matchup_win_rate','home_game','won'])]
        all_teams_last_match = data.loc[data.season_year == this_year, ['team']+non_calc_columns]\
            .groupby('team')\
            .last()\
            .reset_index()

        # Add tournament seed
        all_teams_last_match = all_teams_last_match.merge(teams[['team','seed']], on='team', how='left')

        if not first_four_complete:
            # Predict first_four outcome
            first_four_teams = teams.loc[teams.first_four == True]
            first_four_matchups = tourney.get_first_four_matches(first_four_teams)

            # build matchup metrics
            team1s = list(first_four_matchups.keys())
            team2s = list(first_four_matchups.values())
            ff_team1_metrics = all_teams_last_match.loc[all_teams_last_match.team.isin(team1s)]
            ff_team1_metrics['opponent'] = ff_team1_metrics.team.apply(lambda x:first_four_matchups[x])

            # add team metrics of opponent
            rename_cols = {col:col+'_opp' for col in ff_team1_metrics.columns[1:].tolist()}
            rename_cols['team'] = 'opponent'

            ff_team2_metrics = all_teams_last_match.loc[all_teams_last_match.team.isin(team2s)].rename(columns=rename_cols)
            ff_team_metrics = ff_team1_metrics.merge(ff_team2_metrics, on='opponent', how='inner')

            # Calculate additional metrics (underdog, fan-favorite, matchup-win-rate)
            ff_team_metrics = tourney.calculate_tournament_underdog(ff_team_metrics)
            ff_team_metrics = tourney.calculate_tournament_fan_favorite(ff_team_metrics)
            ff_team_metrics = tourney.calculate_tournament_matchup_win_rate(
                [list(x) for x in first_four_matchups.items()],
                ff_team_metrics,
                data)

            # Strip out team names and seedings
            qual_cols = ['team','opponent','seed','seed_opp']
            ff_team_metrics_qualitative_cols = ff_team_metrics[qual_cols]
            ff_team_metrics.drop(qual_cols, axis=1, inplace=True)

            # Order columns the same as df_features
            ff_team_metrics = ff_team_metrics[df_features.columns[1:]]

            # Get match outcomes
            ff_team_metrics.insert(0, 'win_probability', model.predict_proba(ff_team_metrics)[:, 1])
            ff_team_metrics.insert(0,'win',model.predict(ff_team_metrics.drop('win_probability',axis=1)))

            # Add qualitative columns back to dataset
            ff_final = ff_team_metrics_qualitative_cols.join(ff_team_metrics)
            ff_final.insert(4, 'game_round', 'first_four')
            first_four_losers = ff_final.loc[ff_final.win == 1,'opponent'].values.tolist() +\
                                ff_final.loc[ff_final.win == 0,'team'].values.tolist()
            teams = teams.loc[~teams.team.isin(first_four_losers)]

        else:
            teams = teams.loc[~teams.team.isin(first_four_losers)]

        # Play Remaining tournament
        # Get matchups in the first round ordered by bracket regions (ensures teams play in correct order)
        first_round_matchups = tourney.get_first_round_matches(teams)

        # Run remaining tournament for a single simulation
        rnd_results = tourney.run_tournament(
            first_round_matchups, all_teams_last_match, data, model, non_calc_columns, df_features)

        if not first_four_complete:
            final_output = pd.concat([ff_final,rnd_results])
        else:
            final_output = rnd_results.copy()

        # Calculate qualitative columns to permit monte-carlo win frequency analysis
        final_output.insert(0,'winning_team',final_output.apply(lambda x:x.team if x.win == 1 else x.opponent, axis=1))
        final_output.reset_index(inplace=True)
        final_output.rename(columns={'index':'game_number_in_round'}, inplace=True)
        final_output.insert(0,'game_slot',
                            final_output.apply(lambda x:x.game_round+'_'+str(x.game_number_in_round+1),axis=1))

        # Append win counts for each game_slot per simulation
        if i == 0:
            win_counts = {slot: {team: 1} for slot, team in final_output[['game_slot','winning_team']].values}
        else:
            tmp_results = {slot: team for slot, team in final_output[['game_slot','winning_team']].values}
            for slot, team in tmp_results.items():
                if team in win_counts[slot].keys():
                    win_counts[slot][team] += 1
                else:
                    win_counts[slot][team] = 1

    if (i+1) % (num_simulations // 10) == 0:
        print(f'{i+1} simulations complete')

# Calculate win percentage
for slot in win_counts.keys():
    for team in win_counts[slot].keys():
        win_counts[slot][team] /= num_simulations

# Get winning and runner-up teams and their win percentages
results = {}
for slot in win_counts.keys():
    outcomes = sorted(win_counts[slot].items(), key=lambda x: x[1], reverse=True)
    winner = outcomes[0]
    if len(outcomes) > 1:
        runner_up = outcomes[1]
    else:
        runner_up = (np.nan, np.nan)

    results[slot] = [winner[0], runner_up[0], winner[1], runner_up[1]]

results_df = pd.DataFrame(
    results,
    index=['winning_team','runner_up_team','winning_team_win_perc','runner_up_win_perc']).T

# Log final results
final_output.to_csv(f'data/single_simulation_predictions_{branch}{str(this_year)[-2:]}.csv', index=False)
prediction_filepath = f'data/tournament_predictions_{num_simulations}sims_{branch}{str(this_year)[-2:]}.csv'
results_df.index.name = 'game_slot'
results_df.to_csv(prediction_filepath)

print(f'Simulation Complete\n Results may be found at {prediction_filepath}')

