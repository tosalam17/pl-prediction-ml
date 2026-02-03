import pandas as pd
import understatapi
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

client = understatapi.UnderstatClient()

class Footy_Predictor:
    def __init__(self, roll_n = 5, min_n = 1):
        self.roll_n = roll_n
        self.min_n = min_n
        self.eps = 1 * 10**(-6)
        self.team_map = {
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Nott'm Forest": "Nottingham Forest",
            "Newcastle": "Newcastle United",
            "Wolves": "Wolverhampton Wanderers"
        }




        years = ["2021", "2022", "2023", "2024", "2025"]

        seasons = []

        for y in years:
            season_data = client.league("EPL").get_team_data(season = y)
            seasons.append(season_data)

        season_dfs = []
        for s in seasons:
            rows = []
            for team_id, team_data in s.items():
                team_name = team_data["title"]

                for match in team_data["history"]:
                    row = match.copy()
                    row["team"] = team_name
                    row["team_id"] = team_id
                    rows.append(row)
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values(["team", "date"])
            df = df.set_index(["team", "date"])

            #new columns
            df['deep_ratio'] = round(df['deep'] / (df['deep'] + df['deep_allowed'] + self.eps),2)


            df = self.extract_ppda(df, col = "ppda", prefix = "ppda")
            df = self.extract_ppda(df, col = "ppda_allowed", prefix = "ppda_allowed")
            

            season_dfs.append(df)

        #the columns we're going to engineer new features on
        mean_metrics = ["xG", "xGA", "npxGD", "npxG", "npxGA", "xpts", "deep_ratio"]

        #the columns I'll want to keep (to later train our model on)
        self.key_stats = [
        'ppda_att', 'ppda_def', 'ppda_allowed_att', 'ppda_allowed_def',
        'rolling_xG_5', 'rolling_xGA_5', 'rolling_npxGD_5', 'rolling_npxG_5', 
        'rolling_npxGA_5', 'rolling_xpts_5', 'rolling_deep_ratio_5', 'rolling_xG_std_5', 
        'rolling_xGA_std_5', 'rolling_npxGD_std_5'
        ]
        for i, s in enumerate(season_dfs):
            for m_col in mean_metrics:
                s = self.rolling_mean(s, m_col)

            for sd_col in ["xG", "xGA", "npxGD"]:
                s = self.rolling_std(s, sd_col)
            s = s.reset_index()[["team", "date"] + self.key_stats]
            s_clean = s.set_index(["team", "date"])
            s_clean = s_clean.dropna()
            

            season_dfs[i] = s_clean
        self.seasons = dict(zip(years, season_dfs))


        game_results = []
        keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        for y in years:
            df = pd.read_csv(f'{y}.csv')[keep_cols]

            #changing all the team names to match our understat team names
            df["HomeTeam"] = df["HomeTeam"].replace(self.team_map)
            df = df.apply(pd.to_numeric, errors="ignore")
            # make label: 1 = home win, 0 = draw, -1 = away win
            df["FTR"] = np.select(
                [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]],
                [1, 0],
                default=-1
            )
            drop_cols = ["FTHG", "FTAG"]
            df = df.drop(columns = drop_cols)
            #change our dates to the proper format
            df["Date"] = pd.to_datetime(df["Date"], dayfirst= True, errors = "raise").dt.date
            df = df.sort_values("Date")


            #drop Arsenals first game so that our results dataframes align with our stats dataframes
            first_game_mask = ((df["HomeTeam"] == "Arsenal") | (df["AwayTeam"] == "Arsenal"))
            first_ars_index = df.loc[first_game_mask].index[:2]
            df = df.drop(index = first_ars_index)

            game_results.append(df)
        self.results = dict(zip(years, game_results))


    

    """ 
    Now it's time to build the methods that we'll use to clean our dfs, 
    engineer new features, merging our dataframes,
    and train some models based on these newfeatures
    """


    def extract_ppda(self, df, col, prefix):
        """
        Extract att/def from Understat PPDA dict column
        and compute the PPDA ratio.
        """
        df[f"{prefix}_att"] = df[col].apply(
            lambda x: x.get("att") if isinstance(x, dict) else np.nan
        )
        df[f"{prefix}_def"] = df[col].apply(
            lambda x: x.get("def") if isinstance(x, dict) else np.nan
        )

        df[prefix] = df[f"{prefix}_att"] / (df[f"{prefix}_def"]+ self.eps)
        return df 


    def rolling_mean(self, df, col):
        """
        A function used to get the rolling mean of a column over a number n amount of games 
        """
        out_col = f'rolling_{col}_{self.roll_n}'
        df[out_col] = (
            df.groupby("team")[f'{col}'].shift(1)
            .rolling(window = self.roll_n, min_periods = self.min_n).mean()
        )

        return df
    
    def rolling_std(self, df, col):
        """
        A function used to get the rolling mean of a column over a number n amount of games 
        """

        out_col = f'rolling_{col}_std_{self.roll_n}'
        df[out_col] = (
            df.groupby("team")[f'{col}'].shift(1)
            .rolling(window = self.roll_n, min_periods = self.min_n).std()
        )
        return df




    def build_match_df(self, year):

        """ 
        Merging our dataframes in order to have a match ready dataframe that we can use our 
        engineered features to train a model and make predictions
        """
        stats = self.seasons[year].reset_index()
        results = self.results[year].copy()

        home = stats.rename(columns={c: f"home_{c}" for c in self.key_stats})
        away = stats.rename(columns={c: f"away_{c}" for c in self.key_stats})

        df = (
            results.merge(
                home, 
                left_on = ["HomeTeam", "Date"],
                right_on = ["team", "date"],
                how = "inner"
            ).merge(
                away,
                left_on = ["AwayTeam", "Date"], 
                right_on = ["team", "date"],
                how = "inner"
            )
        )

        return df.drop(columns = ["team_x", "date_x", "team_y", "date_y"])



    def train_model(self):
        #building the merged dataframes that we'll use to train
        df_2021 = self.build_match_df("2021")
        df_2022 = self.build_match_df("2022")
        df_2023 = self.build_match_df("2023")
        df_2024 = self.build_match_df("2024")
        df_2025 = self.build_match_df("2025")


        #set a cutoff data that we can use to split how far into this season we'll train our model on
        cutoff_date = df_2025["Date"].quantile(0.5)

        df_25_train = df_2025[df_2025["Date"] <= cutoff_date]
        df_25_test = df_2025[df_2025["Date"] > cutoff_date]


        #combine all of our training dataframes
        train_df = pd.concat([df_2021, df_2022, df_2023, df_2024, df_25_train], ignore_index= True)

        drop_cols = ["FTR", "Date", "HomeTeam", "AwayTeam"]

        #make all of our training columns numerical and set our target column that we will predict on
        X_train = train_df.drop(columns= drop_cols)
        y_train = train_df["FTR"]

        X_test = df_25_test.drop(columns= drop_cols)
        y_test = df_25_test["FTR"]

        #create a lable map since we're using XGBoost
        lab_map = {-1:0, 0:1, 1:2}
        y_train = y_train.map(lab_map)
        y_test = y_test.map(lab_map)

        model = XGBClassifier(
            objective= "multi:softprob",
            num_class = 3,
            eval_metric = "mlogloss",
            random_state = 42
        )

        param_grid = {
            "n_estimators": [300, 500, 750],
            "max_depth": [3,5,7],
            "learning_rate": [0.01, 0.03, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree":[0.8, 0.9, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits = 5)

        grid_search = GridSearchCV(
            estimator= model,
            param_grid= param_grid,
            scoring = "accuracy",
            cv = tscv,
            verbose = 0,
            n_jobs = -1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_


        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)

        print(probs.mean(axis = 0))

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        print("Accuracy:", acc)
        print("Log loss:", ll)
        print("\nClassification report:\n", classification_report(y_test, preds, labels=[0,1,2], zero_division=0))
        print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

        # store stuff on self so you can reuse later
        self.best_model = best_model
        self.cutoff_date_2025 = cutoff_date

        # return if you want to use outside the class
        return self.best_model
    
    def get_stats(self, team, date, year):
        stats_df = self.seasons[year]
        stats_df.index = stats_df.index.set_levels(
            pd.to_datetime(stats_df.index.levels[1]),
            level='date')

        cutoff = pd.to_datetime(date)

        team_df = (
            stats_df
            .xs(team, level="team")
            .sort_index()
        )

        team_stats = team_df.loc[:cutoff]

        if team_stats.empty:
            raise ValueError(f"No stats available for {team} before {cutoff}")

        return team_stats.iloc[-1]


    def predict_game(self, home_team, away_team, date, year):
        if not hasattr(self, "best_model"):
            raise ValueError("Model not trained. Call train_model() first.")

        # get pre-match stats
        home_stats = self.get_stats(home_team, date, year)
        away_stats = self.get_stats(away_team, date, year)

        # build feature row
        row = pd.concat([
            home_stats.add_prefix("home_"),
            away_stats.add_prefix("away_")
        ])

        # drop non-feature columns (safety)
        drop_cols = ["FTR", "Date", "HomeTeam", "AwayTeam"]
        row = row.drop([c for c in drop_cols if c in row.index])

        X = row.to_frame().T

        # predict
        pred = self.best_model.predict(X)[0]
        probs = self.best_model.predict_proba(X)[0]

        inv_map = {0: f'{away_team} wins🫣', 1: 'A draw is on the cards😴', 2: f'{home_team} wins🥳'}
        result = inv_map[pred]

        return {
            "prediction": result,
            f"Chance of {home_team} winning": f"{round(float(probs[2]) * 100,2)}%",
            f"Chance of a draw": f"{round(float(probs[1]) * 100, 2)}%",
            f"Chance of {away_team} winning": f"{round(float(probs[0]) * 100, 2)}%"
        }
