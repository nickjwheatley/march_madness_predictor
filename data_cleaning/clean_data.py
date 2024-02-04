#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

# In[3]:


def clean_team_name(teams):
    teams = teams.str.strip()
    expressions = {
        'st\.$':'State',
        '(st\.) \(':'State (',
        '^st\.':'Saint',
        'Ark\.':'Arkansas',
        'U\.':'University',
        'Univ\.':'University',
        'Mt\.':'Mount',
        'Ala\.':'Alabama',
        'Ariz\.':'Arizona',
        'Colo\.':'Colorado',
        'Conn\.':'Connecticut',
        'Fla\.':'Florida',
        'Ga\.':'Georgia',
        'Ill\.':'Illinois',
        'Ind\.':'Indiana',
        'Ky\.':'Kentucky',
        'La\.':'Louisiana',
        'Me\.':'Maine',
        '^NC':'North Carolina',
        'N\.C\.':'North Carolina',
        'N\.M\.':'New Mexico',
        'Okla\.':'Oklahoma',
        'Ore\.':'Oregon',
        'Mich\.':'Michigan',
        'Minn\.':'Minnesota',
        'Miss\.':'Mississippi',
        'Mo\.':'Missouri',
        'Tenn\.':'Tennessee',
        'Tex\.':'Texas',
        'Va\.':'Virginia',
        'Wash\.':'Washington',
        'Wis\.':'Wisconsin',
        'fran\.':'Francisco',
        'Christ\.':'Christian',
        'Cal St\.':'Cal State',
        'Col\.':'College',
        'So\.':'Southern',
        '^A\&M Corpus Christi$':'Texas A&M Corpus Christi',
        'Texas A&M Corpus Chris$':'Texas A&M Corpus Christi',
        'alcorn$':'Alcorn State',
        'army$':'Army West Point',
        'bowling green$':'Bowling Green State',
        'lamar$':'Lamar University',
        ' Maryland$':' (MD)',
        'mcneese$':'McNeese State',
        'val\.':'Valley',
        'ole miss$':'Mississippi',
        'uni$':'Northern Iowa',
        'penn$':'Pennsylvania',
        'Saint Thomas$':'Saint Thomas (FL)',
        'Caro\.$':'Carolina',
        'Int\'l':'International',
        ' Lafayette$':'',
        'Central Connecticut$':'Central Connecticut State',
        'Detroit$':'Detroit Mercy',
        '^Fort Wayne$':'Purdue Fort Wayne',
        'ETSU$': 'East Tennessee State',
        '^Albany$': 'Albany (NY)',
        'Cal ': 'California ',
        'Grambling$': 'Grambling State',
        'Mississippi Valley$':'Mississippi Valley State',
        '^Omaha$':'Nebraska Omaha',
        'Nicholls$':'Nicholls State',
        'NUI$':'Northern Illinois',
        'Prairie View$':'Prairie View A&M',
        '^Queens$':'Queens (NC)',
        'SIUE':'SIU Edwardsville',
        'Saint Francis PA':'Saint Francis (PA)',
        "^Saint John's$":"Saint John's (NY)",
        "Saint Mary's$":"Saint Mary's (CA)",
        ' U$':'',
        '^U ':'',
        'Southern Miss$':'Southern Mississippi',
        'SFA':'Stephen F. Austin',
        '^A\&M Corpus Christi$':'Texas A&M Corpus Christi',
        '^Kansas City':'UMKC',
        'UNCW':'UNC Wilmington',
        '^USC$':'Southern California',
        'UT Rio Grande Valley':'Texas Rio Grande Valley',
        'LIU$': 'LIU Brooklyn',
        '^NIU$':'Northern Illinois',
        'Sam Houston$':'Sam Houston State',
        'SMU$': 'Southern Methodist'
    }
    for exp in expressions.keys():
        teams = teams.str.replace(exp,expressions[exp],case=False,regex=True)
        
    replacements = {
        '-':' ',
        'App ':'Appalachian ',
        'BYU':'Brigham Young',
        'UNLV':'Nevada Las Vegas',
        'FGCU':'Florida Gulf Coast',
        'UConn':'Connecticut',
        'UCF':'Central Florida',
        'UIndy':'Indiana',
        'UIC':'Illinois Chicago',
        'ULM':'Louisiana Monroe',
        'VCU':'Virginia Commonwealth',
        'LSU':'Louisiana State',
        'USC$':'Southern California',
        'USC Upstate':'South Carolina Upstate',
        'Loyola \(IL\)':'Loyola Chicago',
        "Saint Mary\'s$":"Saint Mary's (CA)",
        'Miami FL':'Miami (FL)',
        'UMBC':'Maryland Baltimore County',
        'UMass Lowell':'Massachusetts Lowell',
        'Saint Francis Brooklyn':'Saint Francis (NY)',
        'Seattle U$':'Seattle',
        ' University$':'',
        ' University':'',
        'Loyola Marymount$':'Loyola Marymount (CA)',
        'LMU \(CA\)$':'Loyola Marymount (CA)',
        'UMES':'Maryland Eastern Shore',
        'Mississippi Valley$':'Mississippi Valley State',
        'UT Martin':'Tennessee Martin',
        'UAlbany':'Albany (NY)',
        'UTRGV':'Texas Rio Grande Valley',
        'FIU':'Florida International',
        'FDU':'Fairleigh Dickinson',
        'UIW':'Incarnate Word',
        'Loyola MD':'Loyola (MD)',
        'LMU (CA)':'Loyola Marymount',
        'CSU Bakersfield': "California State Bakersfield",
        'CSUN':'California State Northridge',
        'Miami OH':'Miami (OH)',
        'Loyola (IL)': 'Loyola Chicago',
        'Long Island': 'LIU Brooklyn'
    }
    for r in replacements:
        teams = teams.str.replace(r,replacements[r])
        
    return teams


# In[4]:


def find_inconsistent_team_naming(ncaa,sportsref,bart = None, qualifers_f=None):
    # Find inconsistent team naming
    ncaa_teams = set(list(ncaa.away_team.unique()) + list(ncaa.home_team.unique()))
    if bart is not None:
        for team in sorted(bart.team.unique()):
            if team not in ncaa_teams:
                print(f'bart: {team} not in NCAA team list')
            if team not in sportsref.team.unique():
                print(f'bart: {team} not in Sports-Reference team list')

    for team in sorted(sportsref.team.unique()):
        if team not in ncaa_teams:
            print(f'sports-reference: {team} not in NCAA team list')
            
    if qualifers_f is not None:
        for team in sorted(qualifers_f.team.unique()):
            if team not in ncaa_teams:
                print(f'sports-reference-female-qualifier: {team} not in NCAA team list')
    return


# In[5]:


def clean_data(datas,this_year,branch):
    """
    Cleans extracted data from NCAA, Barttorvik, and Sports-Reference sites for future model
    :param extraction_years: list of season-end years to pull from the websites
    :param season_dates: pandas DataFrame of start, tournament, and end dates of all season years
    :param this_year: int indicating the current season_year for the model to predict
    :param lookback_years: int indicating how many years of data to consider in the model
    :param branch: string indicating whether to work with men or women data
    :return pandas DataFrame of cleaned data ready to train
    """
    # Extract all data from data_extraction notebook
    if branch == 'men':
        ncaa,sportsref,bart = datas
    else:
        ncaa,sportsref = datas
    
    # Clean team names to match across data sources
    ncaa.away_team = clean_team_name(ncaa.away_team)
    ncaa.home_team = clean_team_name(ncaa.home_team)
    sportsref.team = clean_team_name(sportsref.team)
    
    if branch == 'men':
        bart.team = clean_team_name(bart.team)
    
    # Manually assign A&M - couldn't get it to change in previous function
    ncaa.home_team = ncaa.home_team.str.replace('A&M Corpus Christi','Texas A&M Corpus Christi')
    ncaa.away_team = ncaa.away_team.str.replace('A&M Corpus Christi','Texas A&M Corpus Christi')
    
    # Clean names of female ncaa qualifier teams
    if branch == 'women':
        qualifiers_f = pd.read_csv(f'data/ncaa_qualifiers_w{str(this_year)[-2:]}.csv')
        qualifiers_f.team = clean_team_name(qualifiers_f.team)
        qualifiers_f.to_csv(
            f'data/tournament_regions_women{str(this_year)[-2:]}.csv',index=False)
        
        # Print team names that don't match between data sources
        find_inconsistent_team_naming(ncaa,sportsref,qualifers_f=qualifiers_f)
    else:
        find_inconsistent_team_naming(ncaa,sportsref,bart=bart)
    
    
    # Check for duplicates in sportsref
    sportsref_this_year = sportsref.loc[sportsref.season_year == this_year,'team'].copy()
    print(f'sportsref duplicates: {sportsref_this_year[sportsref_this_year.duplicated()].tolist()}')

    # Check for duplicates in bart
    if branch == 'men':
        if this_year in bart.season_year.tolist():
            bart_this_year = bart.loc[bart.season_year == this_year,'team'].copy()
            print(f'bart duplicates: {bart_this_year[bart_this_year.duplicated()].tolist()}')
        
    # Remove Nans
    ncaa.dropna(subset=['home_score','away_score'],inplace=True)
    sportsref.dropna(subset=['sos','pace','pace_opp'],inplace=True)
    
    # Correct sports-reference data types
    float_cols = {}
    int_cols = {}
    for col in sportsref.columns[~sportsref.columns.isin(['team','season_year'])].tolist():
        float_cols[col] = float
        if not (('%' in col) | (col in ['srs','sos'])):
            int_cols[col] = int
    
    for convert in [float_cols,int_cols]:
        sportsref = sportsref.astype(convert)
    
    # Create additional metrics
    ncaa['home_game'] = 0
    ncaa['underdog'] = 0
    ncaa['underdog_opp'] = 0
    
    # Melt home and away teams into one tabular DataFrame
    n1_rename={
        'home_team':'team',
        'away_team':'opponent',
        'home_score':'team_score',
        'away_score':'opponent_score',
        'home_rank':'team_rank',
        'away_rank':'opponent_rank'
    }

    n2_rename={
        'home_team':'opponent',
        'away_team':'team',
        'home_score':'opponent_score',
        'away_score':'team_score',
        'home_rank':'opponent_rank',
        'away_rank':'team_rank'
    }
    n1 = ncaa.rename(columns=n1_rename)
    
    ## Identify home_games
    n1.loc[n1.season_type == 'regular_season','home_game'] = 1

    n2 = ncaa.rename(columns=n2_rename)

    ncaa2 = pd.concat([n1,n2])

    # Flag winning and losing events
    ncaa2.insert(2,'won',0)
    ncaa2.loc[ncaa2.team_score > ncaa2.opponent_score,'won'] = 1

    # Add underdog advantage
    ncaa2.loc[
        (ncaa2.team_rank > ncaa2.opponent_rank) | 
        (np.isnan(ncaa2.team_rank) & ~np.isnan(ncaa2.opponent_rank)),'underdog'] = 1
    
    ncaa2.loc[
        (ncaa2.opponent_rank > ncaa2.team_rank) | 
        (np.isnan(ncaa2.opponent_rank) & ~np.isnan(ncaa2.team_rank)),'underdog_opp'] = 1

    # Add fan_favorite - home team in regular season or underdog in tournaments
    ncaa2['fan_favorite'] = 0
    ncaa2['fan_favorite_opp'] = 0
    ncaa2.loc[
        ((ncaa2.home_game == 1) & (ncaa2.season_type == 'regular_season')) |
        ((ncaa2.underdog == 1) & (ncaa2.season_type == 'ncaa_tournament')),
        'fan_favorite'] = 1
    ncaa2.loc[
        ((ncaa2.home_game == 0) & (ncaa2.season_type == 'regular_season')) |
        ((ncaa2.underdog_opp == 1) & (ncaa2.season_type == 'ncaa_tournament')),
        'fan_favorite_opp'] = 1
    ncaa2.reset_index(drop=True,inplace=True)
    ncaa2.to_csv(f'data/cleaned_ncaa_{branch}.csv',index=False)
    sportsref.to_csv(f'data/cleaned_sportsref_{branch}.csv',index=False)
    if branch == 'men':
        bart.to_csv('data/cleaned_bart.csv',index=False)
        return ncaa2,sportsref,bart
    else:
        return ncaa2,sportsref


# In[ ]:




