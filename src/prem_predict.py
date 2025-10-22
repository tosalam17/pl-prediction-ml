import pandas as pd

class Prem_Predictor:
    def __init__(self, url="https://fbref.com/en/comps/9/Premier-League-Stats", 
                 old_url = 'https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures'):
        self.url = url
        self.old_url = old_url
        self.tables = {}

    def fetch_table(self, table_name, table_link):
        html = pd.read_html(table_link, header=0)
        return html[0]

    def fetch_team_tables(self):
        # Example of scraping team related tables
        teams_url = f"{self.url}#all_stats_standard"
        standard_stats = self.fetch_table('Standard Stats', teams_url)
        shooting_stats = self.fetch_table('Shooting Stats', teams_url.replace('standard', 'shooting'))
        passing_stats = self.fetch_table('Passing Stats', teams_url.replace('standard', 'passing'))
        defense_stats = self.fetch_table('Defense Stats', teams_url.replace('standard', 'defense'))

        self.tables['standard'] = standard_stats
        self.tables['shooting'] = shooting_stats
        self.tables['passing'] = passing_stats
        self.tables['defense'] = defense_stats

    def fetch_player_tables(self):
        # Example of scraping player related tables
        players_url = f"{self.url}#all_stats_standard"
        standard_stats = self.fetch_table('Player Standard Stats', players_url)
        shooting_stats = self.fetch_table('Player Shooting Stats', players_url.replace('standard', 'shooting'))
        passing_stats = self.fetch_table('Player Passing Stats', players_url.replace('standard', 'passing'))
        defense_stats = self.fetch_table('Player Defense Stats', players_url.replace('standard', 'defense'))

        self.tables['player_standard'] = standard_stats
        self.tables['player_shooting'] = shooting_stats
        self.tables['player_passing'] = passing_stats
        self.tables['player_defense'] = defense_stats

    def clean_data(self):
        # Example cleaning function
        for key, df in self.tables.items():
            df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            df.fillna(0, inplace=True)
            self.tables[key] = df

    def combine_stats(self):
        # Example combining function for team stats
        if all(k in self.tables for k in ['standard', 'shooting', 'passing', 'defense']):
            combined = self.tables['standard'].copy()
            combined = combined.merge(self.tables['shooting'], on='Team', suffixes=('', '_shooting'))
            combined = combined.merge(self.tables['passing'], on='Team', suffixes=('', '_passing'))
            combined = combined.merge(self.tables['defense'], on='Team', suffixes=('', '_defense'))
            self.tables['combined_team'] = combined

    def get_combined_stats(self):
        return self.tables.get('combined_team', pd.DataFrame())

    def update_season(self, new_url):
        # Example function to update URL for new season and refetch data
        self.url = new_url
        self.fetch_team_tables()
        self.fetch_player_tables()
        self.clean_data()
        self.combine_stats()

    def run(self):
        # Example master method to run all processes
        self.fetch_team_tables()
        self.fetch_player_tables()
        self.clean_data()
        self.combine_stats()
        return self.get_combined_stats()
