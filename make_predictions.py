import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from process_data import BaseballDataProcessor
from train_models import BaseballPredictor
from betting_analysis import BettingAnalyzer
from typing import Dict

class BaseballPredictions:
    def __init__(self, betting_api_key: str = None):
        self.predictor = BaseballPredictor()
        self.data_processor = BaseballDataProcessor()
        self.models = {}
        self.load_models()
        if betting_api_key:
            self.betting_analyzer = BettingAnalyzer(betting_api_key)
        else:
            self.betting_analyzer = None
        
    def load_models(self):
        """Load the trained models"""
        print("Loading trained models...")
        self.models['winner'] = joblib.load('models/winner_model.joblib')
        self.models['home_runs'] = joblib.load('models/home_runs_model.joblib')
        self.models['away_runs'] = joblib.load('models/away_runs_model.joblib')
        self.models['hits'] = joblib.load('models/hits_model.joblib')
        
    def prepare_prediction_data(self, game_data, model_type='winner'):
        """Prepare game data for prediction by converting it to the right format."""
        # Create a DataFrame with the game data
        game_df = pd.DataFrame([{
            'game_id': game_data['game_id'],
            'date': game_data['date'],
            'home_team': game_data['home_team'],
            'away_team': game_data['away_team'],
            'venue_id': game_data['venue_id'],
            'temp': game_data['temp'],
            'wind_speed': game_data['wind_speed'],
            'condition': game_data['condition'],
            'home_pitcher_player_id': game_data['home_pitcher']['player_id'],
            'away_pitcher_player_id': game_data['away_pitcher']['player_id']
        }])
        
        # Add batter information
        for i, batter in enumerate(game_data['home_lineup'], 1):
            game_df[f'home_batter{i}_player_id'] = batter['player_id']
        
        for i, batter in enumerate(game_data['away_lineup'], 1):
            game_df[f'away_batter{i}_player_id'] = batter['player_id']
        
        # Process game data using BaseballDataProcessor
        processed_data = self.data_processor.process_game(game_df)
        
        # Normalize temperature (assuming range 0-120°F)
        processed_data['temp_normalized'] = processed_data['temp'] / 120.0
        
        # Normalize wind speed (assuming range 0-50 mph)
        processed_data['wind_speed_normalized'] = processed_data['wind_speed'] / 50.0
        
        # Define feature sets for different models
        feature_sets = {
            'winner': [
                'temp_normalized', 'wind_speed_normalized', 'venue_park_factor',
                'home_team_last_10_wins', 'away_team_last_10_wins', 'days_since_last_matchup',
                'matchup_trend', 'home_team_weighted_runs_scored', 'away_team_weighted_runs_scored',
                'home_team_runs_std', 'away_team_runs_std', 'total_previous_matchups',
                'home_team_wins_against', 'weighted_avg_runs_in_matchup', 'matchup_runs_std'
            ],
            'home_runs': [
                'temp_normalized', 'wind_speed_normalized', 'venue_park_factor',
                'matchup_trend', 'home_team_trend', 'away_team_trend',
                'venue_avg_total_runs', 'venue_std_total_runs',
                'home_team_weighted_runs_scored', 'away_team_weighted_runs_scored',
                'home_team_runs_std', 'away_team_runs_std',
                'weighted_avg_runs_in_matchup', 'matchup_runs_std',
                'home_team_last_10_wins', 'away_team_last_10_wins'
            ],
            'away_runs': [
                'temp_normalized', 'wind_speed_normalized', 'venue_park_factor',
                'matchup_trend', 'home_team_trend', 'away_team_trend',
                'venue_avg_total_runs', 'venue_std_total_runs',
                'home_team_weighted_runs_scored', 'away_team_weighted_runs_scored',
                'home_team_runs_std', 'away_team_runs_std',
                'weighted_avg_runs_in_matchup', 'matchup_runs_std',
                'home_team_last_10_wins', 'away_team_last_10_wins'
            ]
        }
        
        # Select features for the specified model
        expected_features = feature_sets[model_type]
        features = pd.DataFrame()
        
        for feature in expected_features:
            if feature in processed_data.columns:
                features[feature] = processed_data[feature]
            else:
                features[feature] = 0  # Default value for missing features
        
        # Fill any missing values with 0
        features = features.fillna(0)
        
        # Ensure all values are between 0 and 1
        features = features.clip(0, 1)
        
        return features
        
    def _calculate_confidence(self, probability):
        """Calculate confidence score based on probability."""
        # Convert probability to confidence score (0-1)
        if probability < 0.5:
            probability = 1 - probability
        
        # Scale confidence based on how far from 0.5 the probability is
        confidence = (probability - 0.5) * 2
        return round(confidence, 3)

    def predict_game(self, game_data: dict) -> dict:
        """Make predictions for a single game."""
        try:
            # Prepare data for each model
            winner_data = self.prepare_prediction_data(game_data, 'winner')
            home_runs_data = self.prepare_prediction_data(game_data, 'home_runs')
            away_runs_data = self.prepare_prediction_data(game_data, 'away_runs')

            # Make predictions
            home_win_prob = float(self.models['winner'].predict_proba(winner_data)[0][1])
            predicted_home_runs = float(self.models['home_runs'].predict(home_runs_data)[0])
            predicted_away_runs = float(self.models['away_runs'].predict(away_runs_data)[0])
            
            # Calculate confidence
            confidence = self._calculate_confidence(home_win_prob)
            
            # Create prediction dictionary
            prediction = {
                'game_id': game_data.get('game_id', ''),
                'date': game_data.get('date', ''),
                'home_team': game_data.get('home_team', ''),
                'away_team': game_data.get('away_team', ''),
                'home_win_probability': round(home_win_prob * 100, 1),
                'predicted_home_runs': round(predicted_home_runs, 1),
                'predicted_away_runs': round(predicted_away_runs, 1),
                'predicted_total': round(predicted_home_runs + predicted_away_runs, 1),
                'confidence': round(confidence * 100, 1)
            }
            
            return prediction
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                'game_id': game_data.get('game_id', ''),
                'date': game_data.get('date', ''),
                'home_team': game_data.get('home_team', ''),
                'away_team': game_data.get('away_team', ''),
                'error': str(e)
            }
        
    def predict_player_hits(self, player_data):
        """Predict if a player will get a hit in their next game"""
        features = self.prepare_prediction_data(player_data)
        hit_prob = self.models['hits'].predict_proba(features)[0][1]
        
        prediction = {
            'player_name': player_data['player_name'],
            'team': player_data['team'],
            'hit_probability': f"{hit_prob:.1%}"
        }
        
        # Update performance tracking
        if self.betting_analyzer:
            prediction['performance_summary'] = self.betting_analyzer.get_performance_summary()
        
        return prediction
        
    def predict_upcoming_games(self, games_data):
        """Make predictions for multiple upcoming games"""
        predictions = []
        for game in games_data:
            pred = self.predict_game(game)
            predictions.append(pred)
        return predictions
        
    def update_prediction_results(self, game_id: str, actual_results: Dict):
        """Update prediction performance with actual results"""
        if not self.betting_analyzer:
            return
            
        # Update winner prediction
        predicted_winner = actual_results.get('predicted_winner')
        actual_winner = actual_results.get('actual_winner')
        if predicted_winner and actual_winner:
            self.betting_analyzer.update_winner_prediction(predicted_winner, actual_winner)
            
        # Update runs prediction
        predicted_total = actual_results.get('predicted_total')
        actual_total = actual_results.get('actual_total')
        if predicted_total is not None and actual_total is not None:
            self.betting_analyzer.update_runs_prediction(predicted_total, actual_total)
            
        # Update player hit prediction
        player_name = actual_results.get('player_name')
        predicted_hit = actual_results.get('predicted_hit')
        actual_hit = actual_results.get('actual_hit')
        if player_name and predicted_hit is not None and actual_hit is not None:
            self.betting_analyzer.update_hit_prediction(player_name, predicted_hit, actual_hit)
            
        # Save updated performance metrics
        self.betting_analyzer.save_performance_metrics()

def main():
    # Example usage
    api_key = "df8571eddb753cdaa066497b9058722f"
    predictor = BaseballPredictions(betting_api_key=api_key)
    
    # Example game data
    example_game = {
        'date': '2025-04-08',
        'home_team': 'Yankees',
        'away_team': 'Red Sox',
        'venue_id': '3313',
        'temp': 65,
        'wind_speed': 8,
        'condition': 'Partly Cloudy'
    }
    
    # Make prediction
    prediction = predictor.predict_game(example_game)
    print("\nGame Prediction:")
    print(f"{prediction['away_team']} @ {prediction['home_team']}")
    print(f"Date: {prediction['date']}")
    print(f"Home Win Probability: {prediction['home_win_probability']}")
    print(f"Predicted Score: {prediction['away_team']} {prediction['predicted_away_runs']} - {prediction['predicted_home_runs']} {prediction['home_team']}")
    print(f"Predicted Total: {prediction['predicted_total']}")
    
    # Print betting analysis if available
    if 'betting_analysis' in prediction:
        print("\nBetting Analysis:")
        for rec in prediction['betting_recommendations']:
            print(f"\n{rec['type'].upper()} - {rec['team'] if 'team' in rec else f'Line: {rec['line']}'}")
            print(f"Odds: {rec['odds']}")
            print(f"Edge: {rec['edge']:.1%}")
            print(f"Kelly: {rec['kelly']:.1%}")
            print(f"Confidence: {rec['confidence']}")
            
        # Print performance summary
        print("\nPrediction Performance:")
        for metric, value in prediction['performance_summary'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.1%}")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 