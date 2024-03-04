import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from openai import OpenAI
from googlesearch import search
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# data = pd.read_csv('../transformed_data_men_21_to_24.csv')

#provides a sentiment analysis score for a team based on ChatGPT
def sent_data(df):
    teams = list(df['team'].unique())
    team_dict = {}

    client=OpenAI(api_key='')
    total_calls = 0
    for team in teams:
        prompt = f"Provide a strength of team score on a scale of 1-100 for the 2023-2024 {team} Men's basketball team based on their performance and the quality of the teams they have played. Please provide only the score value"
        while True:
            completion = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are a sports analyst skilled in evaluating NCAA teams."},
                    {"role": "user", "content": prompt}
                ]
            )
            total_calls+=1
            sent_score = completion.choices[0].message.content
            print(prompt)
            print('Score: ', sent_score)
            print('Total Calls: ', total_calls)
            if len(sent_score) < 4:
                sent_score = int(sent_score)
                break
            if total_calls > 200: break
        team_dict[team] = sent_score
        if total_calls > 200: 
            print('Too Many Calls!! Fix Something!!')
            print('Total Calls > 200')
            break
    df['sent_score'] = df['team'].map(team_dict)
    print(total_calls)

    return df

#performs logisitc regression based on the specific NCAA dataset
def log_model(df):
    desired_features = ['won', 
        'home_game', 'underdog', 'underdog_opp', 'fan_favorite',
        'fan_favorite_opp', 'de', 'oe', 'te', 'pace', 'physicality_score',
        'sos', 'srs', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta',
        'ft%', 'orb', 'trb', 'ast', 'stl', 'blk', 'pf', 'tov%', 'de_opp',
        'oe_opp', 'te_opp', 'pace_opp', 'physicality_score_opp', 'sos_opp',
        'srs_opp', 'fg_opp', 'fga_opp', 'fg%_opp', '3p_opp', '3pa_opp',
        '3p%_opp', 'ft_opp', 'fta_opp', 'ft%_opp', 'orb_opp', 'trb_opp',
        'ast_opp', 'stl_opp', 'blk_opp', 'pf_opp', 'tov%_opp', 'seed',
        'seed_opp', 'luck', 'luck_opp', 'choke_rate',
        'choke_rate_opp', 'upset_rate', 'upset_rate_opp', 'win_streak',
        'win_streak_opp', 'matchup_win_rate']
    df = df[desired_features].copy()
    df.fillna(0,inplace=True)

    y = df.iloc[:, 0] #selects the first column (wins)

    X = df.iloc[:, 1:] #selects all columns except the win column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)  # Increase max_iter if the model doesn't converge
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


#performs a google search based on a team's school name
def prompt_search(team_name):
    search_query = f"NCAA {team_name} Men's 2024 season basketball strength analysis"
    g_search = search(search_query, advanced = True)
    search_results = ''
    for i, result in enumerate(g_search, start=1):
        search_results = search_results + str(result)
        print(len(search_results))

#perform correlation analysis
def corr_analysis(df):
    desired_features = ['won', 
        'home_game', 'underdog', 'underdog_opp', 'fan_favorite',
        'fan_favorite_opp', 'de', 'oe', 'te', 'pace', 'physicality_score',
        'sos', 'srs', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta',
        'ft%', 'orb', 'trb', 'ast', 'stl', 'blk', 'pf', 'tov%', 'de_opp',
        'oe_opp', 'te_opp', 'pace_opp', 'physicality_score_opp', 'sos_opp',
        'srs_opp', 'fg_opp', 'fga_opp', 'fg%_opp', '3p_opp', '3pa_opp',
        '3p%_opp', 'ft_opp', 'fta_opp', 'ft%_opp', 'orb_opp', 'trb_opp',
        'ast_opp', 'stl_opp', 'blk_opp', 'pf_opp', 'tov%_opp', 'seed',
        'seed_opp', 'luck', 'luck_opp', 'choke_rate',
        'choke_rate_opp', 'upset_rate', 'upset_rate_opp', 'win_streak',
        'win_streak_opp', 'matchup_win_rate']
    df = df[desired_features].copy()
    df.fillna(0,inplace=True)
    correlation_matrix = df.corr(method='pearson')
    plt.figure(figsize=(40, 40))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

def xgboost(df):
    desired_features = ['won', 
        'home_game', 'underdog', 'underdog_opp', 'fan_favorite',
        'fan_favorite_opp', 'de', 'oe', 'te', 'pace', 'physicality_score',
        'sos', 'srs', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta',
        'ft%', 'orb', 'trb', 'ast', 'stl', 'blk', 'pf', 'tov%', 'de_opp',
        'oe_opp', 'te_opp', 'pace_opp', 'physicality_score_opp', 'sos_opp',
        'srs_opp', 'fg_opp', 'fga_opp', 'fg%_opp', '3p_opp', '3pa_opp',
        '3p%_opp', 'ft_opp', 'fta_opp', 'ft%_opp', 'orb_opp', 'trb_opp',
        'ast_opp', 'stl_opp', 'blk_opp', 'pf_opp', 'tov%_opp', 'seed',
        'seed_opp', 'luck', 'luck_opp', 'choke_rate',
        'choke_rate_opp', 'upset_rate', 'upset_rate_opp', 'win_streak',
        'win_streak_opp', 'matchup_win_rate']
    df = df[desired_features].copy()
    df.fillna(0,inplace=True)

    y = df.iloc[:, 0] #selects the first column (wins)

    X = df.iloc[:, 1:] #selects all columns except the win column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_clf = XGBClassifier(objective='binary:logistic', n_estimators=100, seed=42)

    xgb_clf.fit(X_train, y_train)
    
    y_pred = xgb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def build_best_model(data):
    # Split training and test data
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for scaler in [StandardScaler(), MinMaxScaler()]:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        min_samples_leaf=25,
        random_state=42)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Print model performance
    predictions = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    return model

# xgboost(data)


