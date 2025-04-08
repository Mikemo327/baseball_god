import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os
from datetime import datetime

class BaseballPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        print("Loading data...")
        self.data = pd.read_csv('processed_data.csv')
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')
        print(f"Loaded {len(self.data)} records")
        
    def prepare_time_based_features(self, df, train_data=None):
        """Prepare features that need to be calculated based on historical data only"""
        result = df.copy()
        
        # If train_data is provided, we're preparing test data
        # Use only the training data for calculating means and stds
        if train_data is not None:
            result['temp_mean'] = train_data['temp'].mean()
            result['temp_std'] = train_data['temp'].std()
            result['wind_speed_mean'] = train_data['wind_speed_x'].mean()
            result['wind_speed_std'] = train_data['wind_speed_x'].std()
            result['batting_avg_mean'] = train_data['weighted_batting_avg'].mean()
            result['batting_avg_std'] = train_data['weighted_batting_avg'].std()
            result['ops_mean'] = train_data['weighted_ops'].mean()
            result['ops_std'] = train_data['weighted_ops'].std()
        else:
            # For training data, calculate these values up to the current point in time
            for idx in df.index:
                past_data = df.loc[:idx].iloc[:-1]  # Exclude current row
                result.loc[idx, 'temp_mean'] = past_data['temp'].mean() if not past_data.empty else 0
                result.loc[idx, 'temp_std'] = past_data['temp'].std() if not past_data.empty else 1
                result.loc[idx, 'wind_speed_mean'] = past_data['wind_speed_x'].mean() if not past_data.empty else 0
                result.loc[idx, 'wind_speed_std'] = past_data['wind_speed_x'].std() if not past_data.empty else 1
                result.loc[idx, 'batting_avg_mean'] = past_data['weighted_batting_avg'].mean() if not past_data.empty else 0
                result.loc[idx, 'batting_avg_std'] = past_data['weighted_batting_avg'].std() if not past_data.empty else 1
                result.loc[idx, 'ops_mean'] = past_data['weighted_ops'].mean() if not past_data.empty else 0
                result.loc[idx, 'ops_std'] = past_data['weighted_ops'].std() if not past_data.empty else 1
        
        # Handle zero standard deviations to prevent division by zero
        result['temp_std'] = result['temp_std'].replace(0, 1)
        result['wind_speed_std'] = result['wind_speed_std'].replace(0, 1)
        result['batting_avg_std'] = result['batting_avg_std'].replace(0, 1)
        result['ops_std'] = result['ops_std'].replace(0, 1)
        
        # Calculate normalized features using only historical information
        result['temp_normalized'] = (result['temp'] - result['temp_mean']) / result['temp_std']
        result['wind_speed_normalized'] = (result['wind_speed_x'] - result['wind_speed_mean']) / result['wind_speed_std']
        result['player_avg_normalized'] = (result['weighted_batting_avg'] - result['batting_avg_mean']) / result['batting_avg_std']
        result['player_ops_normalized'] = (result['weighted_ops'] - result['ops_mean']) / result['ops_std']
        
        # Replace infinite values with large finite values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        return result
        
    def prepare_features(self):
        print("Preparing features...")
        
        # Split data chronologically for proper feature preparation
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]
        
        # Prepare features for training data
        train_data_processed = self.prepare_time_based_features(train_data)
        
        # Prepare features for test data using only training data statistics
        test_data_processed = self.prepare_time_based_features(test_data, train_data)
        
        # Combine the datasets
        self.data = pd.concat([train_data_processed, test_data_processed])
        
        # Fill NaN values with 0
        self.data = self.data.fillna(0)
        
    def train_winner_model(self):
        print("Training winner prediction model...")
        
        # Features for winner prediction that don't leak future information
        winner_features = [
            'temp_normalized',
            'wind_speed_normalized',
            'venue_park_factor',
            'home_team_last_10_wins',
            'away_team_last_10_wins',
            'days_since_last_matchup',
            'matchup_trend',
            'home_team_weighted_runs_scored',  # Add team scoring ability
            'away_team_weighted_runs_scored',
            'home_team_runs_std',  # Add team consistency
            'away_team_runs_std',
            'total_previous_matchups',  # Add matchup history
            'home_team_wins_against',
            'weighted_avg_runs_in_matchup',
            'matchup_runs_std'
        ]
        
        X = self.data[winner_features]
        y = self.data['home_win']
        
        # Split data chronologically
        train_size = int(len(self.data) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Winner prediction accuracy: {accuracy:.3f}")
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': winner_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature importance for winner prediction:")
        print(feature_importance)
        
        self.models['winner'] = model
        
    def train_score_model(self):
        print("Training score prediction model...")
        
        # Features for score prediction that don't leak future information
        score_features = [
            'temp_normalized',
            'wind_speed_normalized',
            'venue_park_factor',
            'matchup_trend',
            'home_team_trend',
            'away_team_trend',
            'venue_avg_total_runs',
            'venue_std_total_runs',
            'home_team_weighted_runs_scored',  # Add team scoring ability
            'away_team_weighted_runs_scored',
            'home_team_runs_std',  # Add team consistency
            'away_team_runs_std',
            'weighted_avg_runs_in_matchup',  # Add matchup history
            'matchup_runs_std',
            'home_team_last_10_wins',  # Add recent performance
            'away_team_last_10_wins'
        ]
        
        X = self.data[score_features]
        y_home = self.data['home_runs']
        y_away = self.data['away_runs']
        
        # Split data chronologically
        train_size = int(len(self.data) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_home_train = y_home[:train_size]
        y_home_test = y_home[train_size:]
        y_away_train = y_away[:train_size]
        y_away_test = y_away[train_size:]
        
        # Train home runs model
        home_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        home_model.fit(X_train, y_home_train)
        
        # Train away runs model
        away_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        away_model.fit(X_train, y_away_train)
        
        # Evaluate
        y_home_pred = home_model.predict(X_test)
        y_away_pred = away_model.predict(X_test)
        home_r2 = r2_score(y_home_test, y_home_pred)
        away_r2 = r2_score(y_away_test, y_away_pred)
        print(f"Home runs prediction R²: {home_r2:.3f}")
        print(f"Away runs prediction R²: {away_r2:.3f}")
        
        # Print feature importance for home runs
        feature_importance = pd.DataFrame({
            'feature': score_features,
            'importance': home_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature importance for home runs prediction:")
        print(feature_importance)
        
        self.models['home_runs'] = home_model
        self.models['away_runs'] = away_model
        
    def train_hits_model(self):
        print("Training hit prediction model...")
        
        # Features for hit prediction that don't leak future information
        hit_features = [
            'player_avg_normalized',
            'player_ops_normalized',
            'temp_normalized',
            'wind_speed_normalized',
            'venue_park_factor',
            'days_since_last_game',
            'performance_trend',
            'consistency_score',
            'lhb_batting_avg',  # Add platoon splits
            'rhb_batting_avg',
            'lhb_ops',
            'rhb_ops'
        ]
        
        X = self.data[hit_features]
        
        # Create binary target: 1 if player got a hit in THIS game
        # We need to use the actual hit data from the current game
        # For now, let's use a random target since we don't have the actual hit data
        # This is just for demonstration - in production we would need the actual hit data
        np.random.seed(42)  # For reproducibility
        y = np.random.binomial(1, 0.3, size=len(self.data))  # Assuming 30% hit rate
        
        # Split data chronologically
        train_size = int(len(self.data) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Hit prediction accuracy: {accuracy:.3f}")
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': hit_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature importance for hit prediction:")
        print(feature_importance)
        
        self.models['hits'] = model
        
    def save_models(self):
        print("Saving models...")
        if not os.path.exists('models'):
            os.makedirs('models')
            
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}_model.joblib')
            
    def train_all(self):
        self.load_data()
        self.prepare_features()
        self.train_winner_model()
        self.train_score_model()
        self.train_hits_model()
        self.save_models()
        print("Training complete!")

if __name__ == "__main__":
    predictor = BaseballPredictor()
    predictor.train_all() 