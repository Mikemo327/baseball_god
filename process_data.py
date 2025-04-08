import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
from scipy import stats

class BaseballDataProcessor:
    def __init__(self):
        self.games_df = None
        self.player_stats_df = None
        self.processed_data = None
        
    def load_data(self):
        """Load the raw data from CSV files."""
        try:
            self.games_df = pd.read_csv('history/games.csv')
            self.player_stats_df = pd.read_csv('history/player_stats.csv')
            print("Data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_matchup_features(self) -> pd.DataFrame:
        """Create enhanced features based on historical matchups between teams."""
        matchup_features = []
        
        # Sort games by date
        games_sorted = self.games_df.sort_values('date')
        
        for idx, game in games_sorted.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = pd.to_datetime(game['date'])
            
            # Get previous matchups between these teams
            previous_matchups = games_sorted[
                (games_sorted['date'] < game['date']) &
                (((games_sorted['home_team'] == home_team) & (games_sorted['away_team'] == away_team)) |
                 ((games_sorted['home_team'] == away_team) & (games_sorted['away_team'] == home_team)))
            ].tail(10)  # Last 10 matchups
            
            # Calculate weighted matchup features (more recent games weighted higher)
            weights = np.linspace(1, 0.1, len(previous_matchups)) if len(previous_matchups) > 0 else np.array([])
            weights = weights / weights.sum() if len(weights) > 0 else weights
            
            # Calculate matchup features
            features = {
                'game_id': game['game_id'],
                'total_previous_matchups': len(previous_matchups),
                'home_team_wins_against': len(previous_matchups[
                    ((previous_matchups['home_team'] == home_team) & (previous_matchups['home_win'] == 1)) |
                    ((previous_matchups['away_team'] == home_team) & (previous_matchups['home_win'] == 0))
                ]),
                'weighted_avg_runs_in_matchup': np.average(
                    previous_matchups['home_runs'] + previous_matchups['away_runs'],
                    weights=weights
                ) if len(previous_matchups) > 0 else 0,
                'last_matchup_date': previous_matchups['date'].iloc[-1] if len(previous_matchups) > 0 else None,
                'days_since_last_matchup': (game_date - pd.to_datetime(previous_matchups['date'].iloc[-1])).days if len(previous_matchups) > 0 else None,
                'matchup_runs_std': previous_matchups['home_runs'].std() + previous_matchups['away_runs'].std() if len(previous_matchups) > 0 else 0,
                'matchup_trend': self._calculate_trend(previous_matchups['home_runs'] + previous_matchups['away_runs']) if len(previous_matchups) > 0 else 0
            }
            
            matchup_features.append(features)
        
        return pd.DataFrame(matchup_features)

    def create_player_features(self) -> pd.DataFrame:
        """Create enhanced features based on player performance."""
        player_features = []
        
        # Sort player stats by date
        player_stats_sorted = self.player_stats_df.sort_values('date')
        
        for idx, player_stat in player_stats_sorted.iterrows():
            player_id = player_stat['player_id']
            game_date = pd.to_datetime(player_stat['date'])
            
            # Get player's last 10 games
            previous_games = player_stats_sorted[
                (player_stats_sorted['player_id'] == player_id) &
                (player_stats_sorted['date'] < player_stat['date'])
            ].tail(10)
            
            # Calculate weighted player performance features
            weights = np.linspace(1, 0.1, len(previous_games)) if len(previous_games) > 0 else np.array([])
            weights = weights / weights.sum() if len(weights) > 0 else weights
            
            # Calculate player performance features
            features = {
                'game_id': player_stat['game_id'],
                'player_id': player_id,
                'weighted_batting_avg': np.average(previous_games['batting_avg'], weights=weights) if len(previous_games) > 0 else None,
                'weighted_ops': np.average(previous_games['ops'], weights=weights) if len(previous_games) > 0 else None,
                'weighted_slg': np.average(previous_games['slg'], weights=weights) if len(previous_games) > 0 else None,
                'weighted_obp': np.average(previous_games['obp'], weights=weights) if len(previous_games) > 0 else None,
                'last_10_games_hits': previous_games['hits'].sum() if len(previous_games) > 0 else 0,
                'last_10_games_home_runs': previous_games['home_runs'].sum() if len(previous_games) > 0 else 0,
                'last_10_games_rbis': previous_games['rbis'].sum() if len(previous_games) > 0 else 0,
                'days_since_last_game': (game_date - pd.to_datetime(previous_games['date'].iloc[-1])).days if len(previous_games) > 0 else None,
                'performance_trend': self._calculate_trend(previous_games['ops']) if len(previous_games) > 0 else 0,
                'consistency_score': self._calculate_consistency(previous_games['ops']) if len(previous_games) > 0 else 0
            }
            
            player_features.append(features)
        
        return pd.DataFrame(player_features)

    def create_team_features(self) -> pd.DataFrame:
        """Create enhanced features based on team performance."""
        team_features = []
        
        # Sort games by date
        games_sorted = self.games_df.sort_values('date')
        
        for idx, game in games_sorted.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = pd.to_datetime(game['date'])
            
            # Get team's last 10 games
            home_previous_games = games_sorted[
                (games_sorted['date'] < game['date']) &
                ((games_sorted['home_team'] == home_team) | (games_sorted['away_team'] == home_team))
            ].tail(10)
            
            away_previous_games = games_sorted[
                (games_sorted['date'] < game['date']) &
                ((games_sorted['home_team'] == away_team) | (games_sorted['away_team'] == away_team))
            ].tail(10)
            
            # Calculate team runs for previous games
            home_team_runs = home_previous_games.apply(
                lambda x: x['home_runs'] if x['home_team'] == home_team else x['away_runs'], axis=1
            )
            
            away_team_runs = away_previous_games.apply(
                lambda x: x['home_runs'] if x['home_team'] == away_team else x['away_runs'], axis=1
            )
            
            # Calculate weighted team performance features
            home_weights = np.linspace(1, 0.1, len(home_team_runs)) if len(home_team_runs) > 0 else np.array([])
            away_weights = np.linspace(1, 0.1, len(away_team_runs)) if len(away_team_runs) > 0 else np.array([])
            
            if len(home_weights) > 0:
                home_weights = home_weights / home_weights.sum()
            if len(away_weights) > 0:
                away_weights = away_weights / away_weights.sum()
            
            # Calculate team performance features
            features = {
                'game_id': game['game_id'],
                'home_team_last_10_wins': len(home_previous_games[
                    ((home_previous_games['home_team'] == home_team) & (home_previous_games['home_win'] == 1)) |
                    ((home_previous_games['away_team'] == home_team) & (home_previous_games['home_win'] == 0))
                ]),
                'away_team_last_10_wins': len(away_previous_games[
                    ((away_previous_games['home_team'] == away_team) & (away_previous_games['home_win'] == 1)) |
                    ((away_previous_games['away_team'] == away_team) & (away_previous_games['home_win'] == 0))
                ]),
                'home_team_weighted_runs_scored': np.average(home_team_runs, weights=home_weights) if len(home_team_runs) > 0 else 0,
                'away_team_weighted_runs_scored': np.average(away_team_runs, weights=away_weights) if len(away_team_runs) > 0 else 0,
                'home_team_runs_std': home_team_runs.std() if len(home_team_runs) > 0 else 0,
                'away_team_runs_std': away_team_runs.std() if len(away_team_runs) > 0 else 0,
                'home_team_trend': self._calculate_trend(home_team_runs) if len(home_team_runs) > 0 else 0,
                'away_team_trend': self._calculate_trend(away_team_runs) if len(away_team_runs) > 0 else 0
            }
            
            team_features.append(features)
        
        return pd.DataFrame(team_features)

    def create_game_features(self) -> pd.DataFrame:
        """Create enhanced features based on game context."""
        game_features = []
        
        for idx, game in self.games_df.iterrows():
            # Calculate game context features
            features = {
                'game_id': game['game_id'],
                'temperature': game['temp'],
                'wind_speed': game['wind_speed'],
                'condition': game['condition'],
                'venue_id': game['venue_id'],
                'is_roof_closed': game['condition'] == 'Roof Closed',
                'is_night_game': self._is_night_game(game['date']),
                'is_weekend': self._is_weekend(game['date']),
                'days_since_season_start': self._days_since_season_start(game['date']),
                'month': pd.to_datetime(game['date']).month,
                'day_of_week': pd.to_datetime(game['date']).dayofweek
            }
            
            game_features.append(features)
        
        return pd.DataFrame(game_features)

    def create_venue_features(self) -> pd.DataFrame:
        """Create enhanced features based on venue-specific scoring trends."""
        venue_features = []
        
        # Calculate venue-specific stats
        venue_stats = self.games_df.groupby('venue_id').agg({
            'home_runs': ['mean', 'std', 'median'],
            'away_runs': ['mean', 'std', 'median'],
            'game_id': 'count'
        }).reset_index()
        
        venue_stats.columns = ['venue_id', 'avg_home_runs', 'std_home_runs', 'median_home_runs',
                             'avg_away_runs', 'std_away_runs', 'median_away_runs', 'total_games']
        
        # Calculate park factor (runs scored at venue vs league average)
        league_avg_runs = (self.games_df['home_runs'].mean() + self.games_df['away_runs'].mean()) / 2
        venue_stats['park_factor'] = ((venue_stats['avg_home_runs'] + venue_stats['avg_away_runs']) / 2) / league_avg_runs
        
        # Calculate venue-specific weather impact
        venue_weather = self.games_df.groupby('venue_id').agg({
            'temp': ['mean', 'std'],
            'wind_speed': ['mean', 'std']
        }).reset_index()
        
        venue_weather.columns = ['venue_id', 'avg_temp', 'std_temp', 'avg_wind_speed', 'std_wind_speed']
        
        # Merge weather stats with venue stats
        venue_stats = venue_stats.merge(venue_weather, on='venue_id', how='left')
        
        # Add venue features to each game
        for idx, game in self.games_df.iterrows():
            venue_id = game['venue_id']
            venue_data = venue_stats[venue_stats['venue_id'] == venue_id].iloc[0]
            
            features = {
                'game_id': game['game_id'],
                'venue_avg_total_runs': venue_data['avg_home_runs'] + venue_data['avg_away_runs'],
                'venue_std_total_runs': venue_data['std_home_runs'] + venue_data['std_away_runs'],
                'venue_median_total_runs': venue_data['median_home_runs'] + venue_data['median_away_runs'],
                'venue_park_factor': venue_data['park_factor'],
                'venue_games_played': venue_data['total_games'],
                'venue_avg_temp': venue_data['avg_temp'],
                'venue_std_temp': venue_data['std_temp'],
                'venue_avg_wind_speed': venue_data['avg_wind_speed'],
                'venue_std_wind_speed': venue_data['std_wind_speed']
            }
            
            venue_features.append(features)
        
        return pd.DataFrame(venue_features)

    def create_pitcher_batter_features(self) -> pd.DataFrame:
        """Create enhanced features based on pitcher-batter matchups and handedness."""
        pitcher_batter_features = []
        
        # Get pitcher and batter stats from player_stats
        pitcher_stats = self.player_stats_df[self.player_stats_df['position'] == 'P']
        batter_stats = self.player_stats_df[self.player_stats_df['position'] != 'P']
        
        # Calculate LHP/RHP specific stats
        for idx, game in self.games_df.iterrows():
            game_id = game['game_id']
            
            # Get pitchers and batters for this game
            game_pitchers = pitcher_stats[pitcher_stats['game_id'] == game_id]
            game_batters = batter_stats[batter_stats['game_id'] == game_id]
            
            # Calculate LHP/RHP stats
            lhp_stats = game_pitchers[game_pitchers['throw_arm'] == 'L']
            rhp_stats = game_pitchers[game_pitchers['throw_arm'] == 'R']
            
            # Calculate batter stats by handedness
            lhb_stats = game_batters[game_batters['bat_side'] == 'L']
            rhb_stats = game_batters[game_batters['bat_side'] == 'R']
            
            # Calculate platoon advantage metrics
            lhp_vs_lhb = len(lhp_stats) > 0 and len(lhb_stats) > 0
            rhp_vs_rhb = len(rhp_stats) > 0 and len(rhb_stats) > 0
            
            features = {
                'game_id': game_id,
                'lhp_era': lhp_stats['era'].mean() if len(lhp_stats) > 0 else None,
                'rhp_era': rhp_stats['era'].mean() if len(rhp_stats) > 0 else None,
                'lhp_whip': lhp_stats['whip'].mean() if len(lhp_stats) > 0 else None,
                'rhp_whip': rhp_stats['whip'].mean() if len(rhp_stats) > 0 else None,
                'lhp_k_per_9': lhp_stats['k_per_9'].mean() if len(lhp_stats) > 0 else None,
                'rhp_k_per_9': rhp_stats['k_per_9'].mean() if len(rhp_stats) > 0 else None,
                'lhp_bb_per_9': lhp_stats['bb_per_9'].mean() if len(lhp_stats) > 0 else None,
                'rhp_bb_per_9': rhp_stats['bb_per_9'].mean() if len(rhp_stats) > 0 else None,
                'lhb_batting_avg': lhb_stats['batting_avg'].mean() if len(lhb_stats) > 0 else None,
                'rhb_batting_avg': rhb_stats['batting_avg'].mean() if len(rhb_stats) > 0 else None,
                'lhb_ops': lhb_stats['ops'].mean() if len(lhb_stats) > 0 else None,
                'rhb_ops': rhb_stats['ops'].mean() if len(rhb_stats) > 0 else None,
                'lhb_slg': lhb_stats['slg'].mean() if len(lhb_stats) > 0 else None,
                'rhb_slg': rhb_stats['slg'].mean() if len(rhb_stats) > 0 else None,
                'lhb_obp': lhb_stats['obp'].mean() if len(lhb_stats) > 0 else None,
                'rhb_obp': rhb_stats['obp'].mean() if len(rhb_stats) > 0 else None,
                'platoon_advantage_lhp_lhb': lhp_vs_lhb,
                'platoon_advantage_rhp_rhb': rhp_vs_rhb,
                'platoon_advantage_score': self._calculate_platoon_advantage(lhp_stats, rhp_stats, lhb_stats, rhb_stats)
            }
            
            pitcher_batter_features.append(features)
        
        return pd.DataFrame(pitcher_batter_features)

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate the trend of a series using linear regression."""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope

    def _calculate_consistency(self, series: pd.Series) -> float:
        """Calculate the consistency score (inverse of coefficient of variation)."""
        if len(series) < 2 or series.std() == 0:
            return 0
        return series.mean() / series.std()

    def _calculate_platoon_advantage(self, lhp_stats: pd.DataFrame, rhp_stats: pd.DataFrame,
                                   lhb_stats: pd.DataFrame, rhb_stats: pd.DataFrame) -> float:
        """Calculate a platoon advantage score."""
        score = 0
        
        # LHP vs LHB (disadvantage)
        if len(lhp_stats) > 0 and len(lhb_stats) > 0:
            score -= 1
        
        # RHP vs RHB (disadvantage)
        if len(rhp_stats) > 0 and len(rhb_stats) > 0:
            score -= 1
        
        # LHP vs RHB (advantage)
        if len(lhp_stats) > 0 and len(rhb_stats) > 0:
            score += 1
        
        # RHP vs LHB (advantage)
        if len(rhp_stats) > 0 and len(lhb_stats) > 0:
            score += 1
        
        return score

    def _is_night_game(self, date_str: str) -> bool:
        """Determine if a game is likely a night game based on the date."""
        # This is a simplified version - in reality, you'd want to use actual game times
        return True  # Assuming all games are night games for now

    def _is_weekend(self, date_str: str) -> bool:
        """Determine if a game is on a weekend."""
        date = pd.to_datetime(date_str)
        return date.dayofweek >= 5

    def _days_since_season_start(self, date_str: str) -> int:
        """Calculate days since the start of the season."""
        date = pd.to_datetime(date_str)
        season_start = pd.to_datetime(f"{date.year}-04-01")  # Assuming season starts April 1
        return (date - season_start).days

    def merge_all_features(self) -> pd.DataFrame:
        """Merge all feature sets into a final dataset."""
        # Create all feature sets
        matchup_features = self.create_matchup_features()
        player_features = self.create_player_features()
        team_features = self.create_team_features()
        game_features = self.create_game_features()
        venue_features = self.create_venue_features()
        pitcher_batter_features = self.create_pitcher_batter_features()
        
        # Merge with original games data
        final_dataset = self.games_df.merge(matchup_features, on='game_id', how='left')
        final_dataset = final_dataset.merge(team_features, on='game_id', how='left')
        final_dataset = final_dataset.merge(game_features, on='game_id', how='left')
        final_dataset = final_dataset.merge(venue_features, on='game_id', how='left')
        final_dataset = final_dataset.merge(pitcher_batter_features, on='game_id', how='left')
        
        # Merge player features (this will create multiple rows per game, one for each player)
        final_dataset = final_dataset.merge(
            player_features,
            on='game_id',
            how='left'
        )
        
        return final_dataset

    def process_data(self) -> pd.DataFrame:
        """Main method to process all data."""
        print("Loading data...")
        self.load_data()
        
        print("Creating features...")
        self.processed_data = self.merge_all_features()
        
        print("Data processing complete!")
        return self.processed_data

if __name__ == "__main__":
    processor = BaseballDataProcessor()
    processed_data = processor.process_data()
    
    # Save processed data
    processed_data.to_csv('processed_data.csv', index=False)
    print("Processed data saved to processed_data.csv") 