import pandas as pd
import numpy as np

# Underdog
# Fan Favorite
# Matchup Win Rate


def calculate_tournament_fan_favorite(df):
    df['fan_favorite'] = 0
    df['fan_favorite_opp'] = 0

    # Add underdog advantage
    df.loc[df.underdog == 1, 'fan_favorite'] = 1
    df.loc[(df.underdog_opp == 1), 'fan_favorite_opp'] = 1
    return df


def calculate_tournament_underdog(df):
    df['underdog'] = 0
    df['underdog_opp'] = 0

    # Add underdog advantage
    df.loc[(df.seed > df.seed_opp), 'underdog'] = 1
    df.loc[(df.seed_opp > df.seed), 'underdog_opp'] = 1
    return df


def get_first_four_matches(teams):
    matchups = {}
    i = 0
    for team in teams.values:
        for matchup in matchups.items():
            if team[0] in matchup:
                continue
        for opp in teams.values[i+1:]:
            if (team[1] == opp[1]) & (team[2] == opp[2]):
                matchups[team[0]] = opp[0]
                break
        i+=1
    return matchups


def get_first_round_matches(teams):
    conference_regions = ['East', 'West', 'South', 'Midwest']
    round_one_matchups = [
        (1, 16),
        (8, 9),
        (5, 12),
        (4, 13),
        (6, 11),
        (3, 14),
        (7, 10),
        (2, 15)
    ]

    matchups = []
    for region in conference_regions:
        for matchup in round_one_matchups:
            team = teams.loc[
                (teams.region == region) &
                (teams.seed == matchup[0]), 'team'
            ].item()

            opponent = teams.loc[
                (teams.region == region) &
                (teams.seed == matchup[1]), 'team'
            ].item()

            matchups.append([team, opponent, matchup[0], matchup[1]])

    init_teams_df = pd.DataFrame(matchups, columns=[
        'team', 'opponent', 'seed', 'seed_opp'
    ])
    return init_teams_df

def calculate_tournament_matchup_win_rate(matchups, team_metrics, all_data):
    # Calculate matchup win rate
    # print('Calculating matchup win rate for tournament teams')

    match_wl = []
    for match in matchups:
        wins = all_data.loc[
            (all_data.team == match[0]) &
            (all_data.opponent == match[1]), 'won'
        ].sum()
        games = all_data.loc[
            (all_data.team == match[0]) &
            (all_data.opponent == match[1]), 'won'
        ].count()

        # Remove model's ability to predict using 1-off events
        wins -= 1
        games -= 1

        wins = max(wins, 0)
        games = max(games, 0)
        win_rate = np.nan if games == 0 else wins / games
        match_wl.append(match + [win_rate])

    matchups_df = pd.DataFrame(match_wl,columns=['team','opponent','matchup_win_rate'])

    return team_metrics.merge(matchups_df, on=['team','opponent'], how='left')


def get_next_round_matchups(data):
    round_winners = []
    for i in data.index:
        if data.loc[i].win == 1:
            winning_team = data.loc[i].team
            winning_seed = data.loc[i].seed
        else:
            winning_team = data.loc[i].opponent
            winning_seed = data.loc[i].seed_opp
        round_winners.append([winning_team, winning_seed])

    matchups = []
    matchup = []
    i = 0
    for team in round_winners:
        if i % 2 == 0:
            matchup += team
        else:
            matchups.append(matchup + team)
            matchup = []
        i += 1

    next_round = pd.DataFrame(matchups, columns=['team', 'seed', 'opponent', 'seed_opp'])
    return next_round


def run_tournament(first_round_matchups, team_metrics, all_data, model, keep_cols, df_features):
    opp_cols = {}
    for col in keep_cols:
        if col == 'team':
            opp_cols[col] = 'opponent'
        elif col in ['won', 'game_round']:
            continue
        elif col == 'team_rank':
            opp_cols[col] = 'opponent_rank'
        else:
            opp_cols[col] = f'{col}_opp'

    if 'team' not in opp_cols.keys():
        opp_cols['team'] = 'opponent'

    # if 'seed' not in opp_cols.keys():
    #     opp_cols['seed'] = 'seed_opp'

    df_rnds = []
    teams_df = first_round_matchups.copy()
    for rnd in ['first', 'second', 'sweet16', 'elite_eight', 'final_four', 'championship']:

        round_df = teams_df.merge(team_metrics, on=['team','seed'], how='left')
        round_df = round_df.merge(
            team_metrics[list(opp_cols.keys())].rename(columns=opp_cols), on=['opponent'], how='left')

        # Add calculated metrics
        # round_team_metrics = team_metrics.merge(round_df[['team','opponent']], on='team', how='inner')
        round_df = calculate_tournament_matchup_win_rate(
            round_df[['team','opponent']].values.tolist(), round_df, all_data)
        round_df = calculate_tournament_underdog(round_df)
        round_df = calculate_tournament_fan_favorite(round_df)

        # Strip out team names and seedings
        round_df.reset_index(drop=True, inplace=True) #for future join on index
        qual_cols = ['team', 'opponent', 'seed', 'seed_opp']
        round_df_qualitative_cols = round_df[qual_cols]
        round_df.drop(qual_cols, axis=1, inplace=True)

        # Order columns the same as df_features
        round_df = round_df[df_features.columns[1:]]

        # Get match outcomes
        round_df.insert(0, 'win_probability', model.predict_proba(round_df)[:,1])
        round_df.insert(0, 'win', model.predict(round_df.drop('win_probability', axis=1)))

        round_final = round_df_qualitative_cols.join(round_df)
        round_final['game_round'] = rnd

        df_rnds.append(round_final)

        teams_df = get_next_round_matchups(round_final)

    return pd.concat(df_rnds)