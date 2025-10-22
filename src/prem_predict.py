import pandas as pd

class Prem_Predictor:
    def __init__(self, url="https://fbref.com/en/comps/9/Premier-League-Stats", 
                 old_url = 'https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures',
                 main_table_idx=0, tables=None):
        self.url = url
        self.old = old_url
        self.old_tables = tables if tables is not None else self._scrape_tables(old_url)
        self.tables = tables if tables is not None else self._scrape_tables(url)
        self.prem_table = self.tables[main_table_idx].copy()

        # Clean up: flatten column index if MultiIndex
        if isinstance(self.prem_table.columns, pd.MultiIndex):
            self.prem_table.columns = self.prem_table.columns.droplevel(0)
        self.prem_table = self.prem_table.set_index("Squad")

        # Core data
        self.data = self.prem_table[[
            "Pts", "Pts/MP", "GF", "GA", "GD", "xG", "xGA", "Last 5"
        ]].copy()

        self.teams = list(self.data.index)

        squads = self.tables[2]['Unnamed: 0_level_0']['Squad']

        shots_90 = round(self.tables[9]['Standard']['Sh/90'],2)
        poss = round(self.tables[2]['Unnamed: 3_level_0']['Poss'],2)
        pass_cmp = round(self.tables[10]['Total']['Cmp%'],2)
        op_xg = round(self.tables[9]['Expected']['npxG'],2)
        pen_xg = round(self.tables[9]['Expected']['xG'] - self.tables[9]['Expected']['npxG'],2)
        cs_pct = round(self.tables[4]['Performance']['CS%'],2)
        goals_90 = round(self.tables[2]['Per 90 Minutes']['Gls'],2)
        goals_against_90 = keep_save_pct = round(self.tables[4]['Performance']['GA90'],2)
        keep_save_pct = round(self.tables[4]['Performance']['Save%'],2)
        

        # Optional extra stats (filled later)
        self.shots_per_90 = dict(zip(squads, shots_90))
        self.avg_possession = dict(zip(squads, poss))
        self.avg_pass_completion = dict(zip(squads, pass_cmp))
        self.open_play_xg = dict(zip(squads, op_xg))
        self.penalty_xg = dict(zip(squads, pen_xg))
        self.clean_sheet_pct = dict(zip(squads, cs_pct))
        self.goals_per_90 = dict(zip(squads, goals_90))
        self.goals_against_per_90 = dict(zip(squads, goals_against_90))
        self.keeper_save_pct = dict(zip(squads, keep_save_pct))

    # Scrape main stats table
    def _scrape_tables(self, url):
        return pd.read_html(url)

    # Generic subpage scraper
    def _scrape_team_stat_table(self, stat_type):
        """Generic helper to scrape team stat tables by category (e.g. 'shooting', 'passing', 'possession')."""
        url = f'https://fbref.com/en/comps/9/{stat_type}/Premier-League-Stats'
        tables = pd.read_html(url)
        return tables[0]


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
