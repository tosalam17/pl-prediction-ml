import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class Prem_Predictor:
    def __init__(self, url="https://fbref.com/en/comps/9/Premier-League-Stats",
                 old_stats_url = 'https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats', 
                 old_matches_url = 'https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures',
                 main_table_idx=0, tables=None):
        self.url = url
        self.old_tables = tables if tables is not None else self._scrape_tables(old_stats_url)
        self.old_prem_table = self.old_tables[main_table_idx].copy().set_index('Squad')
        self.tables = tables if tables is not None else self._scrape_tables(url)
        self.prem_table = self.tables[main_table_idx].copy()
        self.old_outcomes_df = tables if tables is not None else self._scrape_tables(old_matches_url)[0]

        # Clean up: flatten column index if MultiIndex
        if isinstance(self.prem_table.columns, pd.MultiIndex):
            self.prem_table.columns = self.prem_table.columns.droplevel(0)
        self.prem_table = self.prem_table.set_index("Squad")

        # Core data

        self.old_data = self.old_prem_table[[
            "Pts", "Pts/MP", "GF", "GA", "GD", "xG", "xGA"
        ]].copy()


        self.data = self.prem_table[[
            "Pts", "Pts/MP", "GF", "GA", "GD", "xG", "xGA", "Last 5"
        ]].copy()

        self.teams = list(self.data.index)

        #series of squads in order of our dataframe

        
        self.squads = self.tables[2]['Unnamed: 0_level_0']['Squad']

        #do the same for the old games, some teams got relegated and promoted
        self.old_squads = self.old_tables[2]['Unnamed: 0_level_0']['Squad']

        # access our stats for the old_season
        old_shots_90 = round(self.old_tables[9]['Standard']['Sh/90'],2)
        old_poss = round(self.old_tables[2]['Unnamed: 3_level_0']['Poss'],2)
        old_pass_cmp = round(self.old_tables[10]['Total']['Cmp%'],2)
        old_op_xg = round(self.old_tables[9]['Expected']['npxG'],2)
        old_pen_xg = round(self.old_tables[9]['Expected']['xG'] - self.old_tables[9]['Expected']['npxG'],2)
        old_cs_pct = round(self.old_tables[4]['Performance']['CS%'],2)
        old_goals_90 = round(self.old_tables[2]['Per 90 Minutes']['Gls'],2)
        old_goals_against_90 = keep_save_pct = round(self.old_tables[4]['Performance']['GA90'],2)
        old_keep_save_pct = round(self.old_tables[4]['Performance']['Save%'],2)
        

        #stats that we'll use to predict
        self.old_shots_per_90 = dict(zip(self.old_squads, old_shots_90))
        self.old_avg_possession = dict(zip(self.old_squads, old_poss))
        self.old_avg_pass_completion = dict(zip(self.old_squads, old_pass_cmp))
        self.old_open_play_xg = dict(zip(self.old_squads, old_op_xg))
        self.old_penalty_xg = dict(zip(self.old_squads, old_pen_xg))
        self.old_clean_sheet_pct = dict(zip(self.old_squads, old_cs_pct))
        self.old_goals_per_90 = dict(zip(self.old_squads, old_goals_90))
        self.old_goals_against_per_90 = dict(zip(self.old_squads, old_goals_against_90))
        self.old_keeper_save_pct = dict(zip(self.old_squads, old_keep_save_pct))


        #access our stats for the current season
        shots_90 = round(self.tables[9]['Standard']['Sh/90'],2)
        poss = round(self.tables[2]['Unnamed: 3_level_0']['Poss'],2)
        pass_cmp = round(self.tables[10]['Total']['Cmp%'],2)
        op_xg = round(self.tables[9]['Expected']['npxG'],2)
        pen_xg = round(self.tables[9]['Expected']['xG'] - self.tables[9]['Expected']['npxG'],2)
        cs_pct = round(self.tables[4]['Performance']['CS%'],2)
        goals_90 = round(self.tables[2]['Per 90 Minutes']['Gls'],2)
        goals_against_90 = keep_save_pct = round(self.tables[4]['Performance']['GA90'],2)
        keep_save_pct = round(self.tables[4]['Performance']['Save%'],2)
        

        #stats that we'll use to predict
        self.shots_per_90 = dict(zip(self.squads, shots_90))
        self.avg_possession = dict(zip(self.squads, poss))
        self.avg_pass_completion = dict(zip(self.squads, pass_cmp))
        self.open_play_xg = dict(zip(self.squads, op_xg))
        self.penalty_xg = dict(zip(self.squads, pen_xg))
        self.clean_sheet_pct = dict(zip(self.squads, cs_pct))
        self.goals_per_90 = dict(zip(self.squads, goals_90))
        self.goals_against_per_90 = dict(zip(self.squads, goals_against_90))
        self.keeper_save_pct = dict(zip(self.squads, keep_save_pct))

    # Scrape main stats table
    def _scrape_tables(self, url):
        return pd.read_html(url)


    

    # Generic subpage scraper
    def _scrape_team_stat_table(self, stat_type):
        """Generic helper to scrape team stat tables by category (e.g. 'shooting', 'passing', 'possession')."""
        url = f'https://fbref.com/en/comps/9/{stat_type}/Premier-League-Stats'
        tables = pd.read_html(url)
        return tables[0]


    def get_old_team_stats(self, team):
        base_stats = self.old_data.loc[team].to_dict()
        optional_stats = {
            "Shots Per 90": self.old_shots_per_90.get(team) if self.old_shots_per_90 else None,
            "Avg Possession": self.old_avg_possession.get(team) if self.old_avg_possession else None,
            "Avg Pass Completion": self.old_avg_pass_completion.get(team) if self.old_avg_pass_completion else None,
            "Open Play XG Per 90": self.old_open_play_xg.get(team) if self.old_open_play_xg else None,
            "Penalty XG Per 90": self.old_penalty_xg.get(team) if self.old_penalty_xg else None,
            "Clean Sheet%": self.old_clean_sheet_pct.get(team) if self.old_clean_sheet_pct else None,
            "Goals Per 90": self.old_goals_per_90.get(team) if self.old_goals_per_90 else None,
            "Goals Against Per 90": self.old_goals_against_per_90.get(team) if self.old_goals_against_per_90 else None,
            "Keeper Save%": self.old_keeper_save_pct.get(team) if self.old_keeper_save_pct else None
        }
        stats = {**base_stats, **optional_stats}
      
        return {'Team': team, **stats}
    
 
    #build our match features for the previous season
    def build_old_features(self, home_team, away_team):
        home = self.get_old_team_stats(home_team)
        away = self.get_old_team_stats(away_team)

        features = {
            "Pts/MP_Diff": home["Pts/MP"] - away["Pts/MP"],
            "GF_Diff": home["GF"] - away["GF"],
            "GA_Diff": home["GA"] - away["GA"],
            "GD_Diff": home["GD"] - away["GD"],
            "xG_Diff": home["xG"] - away["xG"],

            "Shots Per 90_Diff": home["Shots Per 90"] - away["Shots Per 90"],
            "Avg Possession_Diff": home["Avg Possession"] - away["Avg Possession"],
            "Avg Pass Completion_Diff": home["Avg Pass Completion"] - away["Avg Pass Completion"],
            "Open Play XG Per 90_Diff": home["Open Play XG Per 90"] - away["Open Play XG Per 90"],
            "Penalty XG Per 90_Diff": home["Penalty XG Per 90"] - away["Penalty XG Per 90"],
            "Clean Sheet%_Diff": home["Clean Sheet%"] - away["Clean Sheet%"],
            "Goals Per 90_Diff": home["Goals Per 90"] - away["Goals Per 90"],
            "Goals Against Per 90_Diff": home["Goals Against Per 90"] - away["Goals Against Per 90"],
            "Keeper Save%_Diff": home["Keeper Save%"] - away["Keeper Save%"],
            "Home Advantage": 1
        }
        total = 0
        for key, value in features.items():
            features[key] = round(value, 2)
            total += features[key]

        features['Total'] = round(total, 2)
        features['Result'] = 1 if features['Total'] >0 else 0 if features['Total'] == 0 else -1

        
        return (features)
    
    # Retrieve a single team's complete stats
    def get_team_stats(self, team):
        base_stats = self.data.loc[team].to_dict()
        optional_stats = {
            "Shots Per 90": self.shots_per_90.get(team) if self.shots_per_90 else None,
            "Avg Possession": self.avg_possession.get(team) if self.avg_possession else None,
            "Avg Pass Completion": self.avg_pass_completion.get(team) if self.avg_pass_completion else None,
            "Open Play XG Per 90": self.open_play_xg.get(team) if self.open_play_xg else None,
            "Penalty XG Per 90": self.penalty_xg.get(team) if self.penalty_xg else None,
            "Clean Sheet%": self.clean_sheet_pct.get(team) if self.clean_sheet_pct else None,
            "Goals Per 90": self.goals_per_90.get(team) if self.goals_per_90 else None,
            "Goals Against Per 90": self.goals_against_per_90.get(team) if self.goals_against_per_90 else None,
            "Keeper Save%": self.keeper_save_pct.get(team) if self.keeper_save_pct else None
        }
        return {**base_stats, **optional_stats}

    # Build match-level features (home vs away)
    def build_match_features(self, home_team, away_team):

        home = self.get_team_stats(home_team)
        away = self.get_team_stats(away_team)

        features = {
            "Pts/MP_Diff": home["Pts/MP"] - away["Pts/MP"],
            "GF_Diff": home["GF"] - away["GF"],
            "GA_Diff": home["GA"] - away["GA"],
            "GD_Diff": home["GD"] - away["GD"],
            "xG_Diff": home["xG"] - away["xG"],

            "Shots Per 90_Diff": home["Shots Per 90"] - away["Shots Per 90"],
            "Avg Possession_Diff": home["Avg Possession"] - away["Avg Possession"],
            "Avg Pass Completion_Diff": home["Avg Pass Completion"] - away["Avg Pass Completion"],
            "Open Play XG Per 90_Diff": home["Open Play XG Per 90"] - away["Open Play XG Per 90"],
            "Penalty XG Per 90_Diff": home["Penalty XG Per 90"] - away["Penalty XG Per 90"],
            "Clean Sheet%_Diff": home["Clean Sheet%"] - away["Clean Sheet%"],
            "Goals Per 90_Diff": home["Goals Per 90"] - away["Goals Per 90"],
            "Goals Against Per 90_Diff": home["Goals Against Per 90"] - away["Goals Against Per 90"],
            "Keeper Save%_Diff": home["Keeper Save%"] - away["Keeper Save%"],
            "Home Advantage": 1
        }
        total = 0
        for key, value in features.items():
            features[key] = round(value, 2)
            total += features[key]

        features['Total'] = round(total, 2)
        features['Result'] = 1 if features['Total'] >0 else 0 if features['Total'] == 0 else -1


        return (features)
    
    def preprocess(self):
        df = self.old_outcomes_df.copy()
        df['Score'] = df['Score'].astype(str).str.replace('–', ' ')
        def nones(x):
            if (str(x).lower().strip() == 'none') or (str(x).lower().strip() == 'nan'):
                x = np.nan

            return x
        
        df = df.apply(nones, axis = 0)
        df.drop('Notes', axis = 1, inplace = True)
        df.dropna(axis = 0, inplace = True )
        
        
        
        
        df[['Home_Goals', 'Away_Goals']] = df['Score'].str.split(' ', expand = True)
        
        df['Home_Goals'] = pd.to_numeric(df['Home_Goals'].str.strip())
        df['Away_Goals'] = pd.to_numeric(df['Away_Goals'].str.strip())

        def result_label(row):
            if row['Home_Goals'] > row['Away_Goals']:
                return 1
            elif row['Home_Goals'] < row['Away_Goals']:
                return -1
            else:
                return 0
        
        df['Result'] = df.apply(result_label, axis = 1)
        df['Goals_Diff'] = df['Home_Goals'] - df['Away_Goals']
        df['Home_Win'] = (df['Result'] == 1).astype(int)

        return df
    
    
        
        

    def train_model(self):
                    
        matches_df = self.preprocess()
        matches_df['Home'] = matches_df['Home'].str.strip()
        matches_df['Away'] = matches_df['Away'].str.strip()
        


        squad_list = list(self.old_squads)
        team_stats = [self.get_old_team_stats(squad) for squad in squad_list]
        

        team_stats_df = pd.DataFrame(team_stats)
        team_stats_df.set_index('Team', inplace = True)
        team_stats_df.index = team_stats_df.index.str.strip()
        
        
        df = (matches_df.merge(team_stats_df.add_prefix('Home_'),
        left_on='Home',
        right_index=True,
        how='left'
    ).merge(team_stats_df.add_prefix('Away_'),
        left_on='Away',
        right_index=True,
        how='left'
    )
)
        feature_cols = [
    'Home_xG', 'Home_xGA', 'Home_Shots Per 90',
    'Home_Avg Possession', 'Home_Avg Pass Completion',
    'Home_Open Play XG Per 90', 'Home_Penalty XG Per 90',
    'Home_Clean Sheet%', 'Home_Goals Per 90', 'Home_Goals Against Per 90',
    'Home_Keeper Save%',
    'Away_xG', 'Away_xGA', 'Away_Shots Per 90',
    'Away_Avg Possession', 'Away_Avg Pass Completion',
    'Away_Open Play XG Per 90', 'Away_Penalty XG Per 90',
    'Away_Clean Sheet%', 'Away_Goals Per 90', 'Away_Goals Against Per 90',
    'Away_Keeper Save%'
]

        
        
        
        
        X = df[feature_cols]
        y = df['Home_Win']


        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

        # Training the model and evaluating
        model = LogisticRegression(max_iter = 5000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model trained. Accuracy: {acc:.3f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        df['Pred_Home_Win_Prob'] = model.predict_proba(X)[:, 1]
        self.model = model
        self.trained_df = df

        

        return model

        
       
