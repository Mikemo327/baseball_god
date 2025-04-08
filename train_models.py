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
        
    def prepare_features(self):
        print("Preparing features...")
        
        # Team performance features
        self.data['home_team_win_pct'] = self.data.groupby('home_team')['home_win'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        self.data['away_team_win_pct'] = self.data.groupby('away_team')['home_win'].transform(
            lambda x: (1 - x).expanding().mean().shift(1)
        )
        
        # Scoring features
        self.data['home_team_runs_avg'] = self.data.groupby('home_team')['home_runs'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        self.data['away_team_runs_avg'] = self.data.groupby('away_team')['away_runs'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Venue features
        self.data['venue_scoring_factor'] = self.data['venue_park_factor']
        
        # Weather features
        self.data['temp_normalized'] = (self.data['temp'] - self.data['temp'].mean()) / self.data['temp'].std()
        self.data['wind_speed_normalized'] = (self.data['wind_speed_x'] - self.data['wind_speed_x'].mean()) / self.data['wind_speed_x'].std()
        
        # Player performance features
        self.data['player_avg_normalized'] = (self.data['weighted_batting_avg'] - self.data['weighted_batting_avg'].mean()) / self.data['weighted_batting_avg'].std()
        self.data['player_ops_normalized'] = (self.data['weighted_ops'] - self.data['weighted_ops'].mean()) / self.data['weighted_ops'].std()
        
        # Fill NaN values with 0
        self.data = self.data.fillna(0)
        
    def train_winner_model(self):
        print("Training winner prediction model...")
        
        # Features for winner prediction
        winner_features = [
            'home_team_win_pct', 'away_team_win_pct',
            'home_team_runs_avg', 'away_team_runs_avg',
            'venue_scoring_factor', 'temp_normalized',
            'wind_speed_normalized', 'home_team_last_10_wins',
            'away_team_last_10_wins'
        ]
        
        X = self.data[winner_features]
        y = self.data['home_win']
        
        # Split data chronologically
        train_size = int(len(self.data) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Winner prediction accuracy: {accuracy:.3f}")
        
        self.models['winner'] = model
        self.scalers['winner'] = scaler
        
    def train_score_model(self):
        print("Training score prediction model...")
        
        # Features for score prediction
        score_features = [
            'home_team_runs_avg', 'away_team_runs_avg',
            'venue_scoring_factor', 'temp_normalized',
            'wind_speed_normalized', 'home_team_weighted_runs_scored',
            'away_team_weighted_runs_scored'
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train home runs model
        home_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        home_model.fit(X_train_scaled, y_home_train)
        
        # Train away runs model
        away_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        away_model.fit(X_train_scaled, y_away_train)
        
        # Evaluate
        y_home_pred = home_model.predict(X_test_scaled)
        y_away_pred = away_model.predict(X_test_scaled)
        home_r2 = r2_score(y_home_test, y_home_pred)
        away_r2 = r2_score(y_away_test, y_away_pred)
        print(f"Home runs prediction R²: {home_r2:.3f}")
        print(f"Away runs prediction R²: {away_r2:.3f}")
        
        self.models['home_runs'] = home_model
        self.models['away_runs'] = away_model
        self.scalers['score'] = scaler
        
    def train_hits_model(self):
        print("Training hit prediction model...")
        
        # Features for hit prediction
        hit_features = [
            'player_avg_normalized', 'player_ops_normalized',
            'temp_normalized', 'wind_speed_normalized',
            'venue_scoring_factor', 'days_since_last_game',
            'lhb_batting_avg', 'rhb_batting_avg',
            'lhb_ops', 'rhb_ops'
        ]
        
        X = self.data[hit_features]
        # Create binary target: 1 if player had a hit in last 10 games, 0 otherwise
        y = (self.data['last_10_games_hits'] > 0).astype(int)
        
        # Split data chronologically
        train_size = int(len(self.data) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Hit prediction accuracy: {accuracy:.3f}")
        
        self.models['hits'] = model
        self.scalers['hits'] = scaler
        
    def save_models(self):
        print("Saving models...")
        if not os.path.exists('models'):
            os.makedirs('models')
            
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}_model.joblib')
            
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'models/{name}_scaler.joblib')
            
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