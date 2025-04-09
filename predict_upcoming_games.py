import pandas as pd
import json
from datetime import datetime
from make_predictions import BaseballPredictions
import os

def load_upcoming_games(csv_path: str) -> list:
    """Load upcoming games from a CSV file."""
    try:
        # Read the CSV file
        games_df = pd.read_csv(csv_path)
        print(f"Loaded {len(games_df)} upcoming games")
        
        # Convert DataFrame rows to list of dictionaries
        games = []
        for _, row in games_df.iterrows():
            game = {
                'game_id': row['game_id'],
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'venue_id': row['venue_id'],
                'temp': row['temp'],
                'wind_speed': row['wind_speed'],
                'condition': row['condition'],
                'home_pitcher': {
                    'player_id': row['home_pitcher_player_id'],
                    'name': row['home_pitcher_name']
                },
                'away_pitcher': {
                    'player_id': row['away_pitcher_player_id'],
                    'name': row['away_pitcher_name']
                },
                'home_lineup': [],
                'away_lineup': []
            }
            
            # Add home lineup
            for i in range(1, 10):
                game['home_lineup'].append({
                    'player_id': row[f'home_batter{i}_player_id'],
                    'name': row[f'home_batter{i}_name'],
                    'position': row[f'home_batter{i}_position'],
                    'order': row[f'home_batter{i}_order']
                })
            
            # Add away lineup
            for i in range(1, 10):
                game['away_lineup'].append({
                    'player_id': row[f'away_batter{i}_player_id'],
                    'name': row[f'away_batter{i}_name'],
                    'position': row[f'away_batter{i}_position'],
                    'order': row[f'away_batter{i}_order']
                })
            
            games.append(game)
        
        return games
    except Exception as e:
        print(f"Error loading upcoming games: {e}")
        return []

def validate_team_name(team_name: str) -> bool:
    """Validate that a team name exists in our historical data"""
    try:
        # Load historical data to check team names
        historical_data = pd.read_csv('history/games.csv')
        valid_teams = set(historical_data['home_team'].unique()) | set(historical_data['away_team'].unique())
        return team_name in valid_teams
    except Exception as e:
        print(f"Error validating team name: {e}")
        return False

def save_predictions(predictions: list, output_path: str):
    """Save predictions to a CSV file."""
    try:
        # Convert predictions to DataFrame
        df = pd.DataFrame(predictions)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def normalize_probability(prob):
    """Normalize a probability value to be between 0 and 1"""
    if isinstance(prob, (int, float)):
        if prob > 1:
            return prob / 100.0
        return max(0, min(1, prob))
    return 0.5

def moneyline_to_probability(moneyline):
    """Convert moneyline odds to implied probability"""
    try:
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)
    except:
        return None

def get_betting_edge(predicted_prob, moneyline):
    """Calculate the edge between predicted probability and implied odds probability"""
    try:
        implied_prob = moneyline_to_probability(float(moneyline))
        if implied_prob is None:
            return None
        return predicted_prob - implied_prob
    except:
        return None

def main():
    """Main function to process upcoming games and make predictions."""
    try:
        # Initialize predictor
        print("Loading trained models...")
        predictor = BaseballPredictions()
        
        # Load upcoming games
        print("\nLoading upcoming games data...")
        games = load_upcoming_games('upcoming_games_template.csv')
        if not games:
            print("No games to process")
            return
        
        print(f"\nProcessing {len(games)} games...")
        
        # Make predictions for each game
        predictions = []
        for game in games:
            print(f"\nAnalyzing game: {game['away_team']} @ {game['home_team']} on {game['date']}")
            print(f"Venue ID: {game['venue_id']}")
            print(f"Weather: {game['temp']}°F, {game['wind_speed']}mph wind, {game['condition']}")
            print(f"Starting Pitchers:")
            print(f"  Home: {game['home_pitcher']['name']} (ID: {game['home_pitcher']['player_id']})")
            print(f"  Away: {game['away_pitcher']['name']} (ID: {game['away_pitcher']['player_id']})")
            
            pred = predictor.predict_game(game)
            if pred:
                # Calculate total runs and ensure all required fields exist
                pred['predicted_home_runs'] = round(pred.get('predicted_home_runs', 0))  # Round to nearest integer
                pred['predicted_away_runs'] = round(pred.get('predicted_away_runs', 0))  # Round to nearest integer
                pred['total_runs'] = pred['predicted_home_runs'] + pred['predicted_away_runs']  # Sum of rounded values
                pred['prediction_confidence'] = pred.get('prediction_confidence', 0.5)  # Default to 50% if missing
                
                # Normalize win probability
                raw_win_prob = pred.get('home_win_probability', 0.5)
                pred['home_win_probability'] = normalize_probability(raw_win_prob)
                
                predictions.append(pred)
                
                print("\nPrediction Details:")
                print(f"Home Win Probability: {pred['home_win_probability']:.1%}")
                print(f"Predicted Score: {pred['away_team']} {pred['predicted_away_runs']} - {pred['predicted_home_runs']} {pred['home_team']}")
                print(f"Total Runs: {pred['total_runs']}")
                print(f"Prediction Confidence: {pred['prediction_confidence']:.1%}")
                
                # Get processed data for additional statistics
                game_df = pd.DataFrame([{
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'venue_id': game['venue_id'],
                    'temp': game['temp'],
                    'wind_speed': game['wind_speed'],
                    'condition': game['condition'],
                    'home_pitcher_player_id': game['home_pitcher']['player_id'],
                    'away_pitcher_player_id': game['away_pitcher']['player_id']
                }])
                
                # Add batter information
                for i, batter in enumerate(game['home_lineup'], 1):
                    game_df[f'home_batter{i}_player_id'] = batter['player_id']
                
                for i, batter in enumerate(game['away_lineup'], 1):
                    game_df[f'away_batter{i}_player_id'] = batter['player_id']
                
                # Process game data
                processed_data = predictor.data_processor.process_game(game_df)
                
                # Print pitcher and batter statistics
                print("\nPitcher Statistics:")
                try:
                    def format_stat(stat_series, format_str='.3f'):
                        if stat_series is None or stat_series.empty or pd.isna(stat_series.iloc[0]):
                            return 'N/A'
                        try:
                            return f"{stat_series.iloc[0]:{format_str}}"
                        except:
                            return 'N/A'
                    
                    print(f"LHP ERA: {format_stat(processed_data.get('lhp_era'))}")
                    print(f"RHP ERA: {format_stat(processed_data.get('rhp_era'))}")
                    print(f"LHP WHIP: {format_stat(processed_data.get('lhp_whip'))}")
                    print(f"RHP WHIP: {format_stat(processed_data.get('rhp_whip'))}")
                    print(f"LHP K/9: {format_stat(processed_data.get('lhp_k_per_9'), '.2f')}")
                    print(f"RHP K/9: {format_stat(processed_data.get('rhp_k_per_9'), '.2f')}")
                except Exception as e:
                    print("Error displaying pitcher statistics:", str(e))
                
                print("\nBatter Statistics:")
                try:
                    print(f"LHB Batting Avg: {format_stat(processed_data.get('lhb_batting_avg'))}")
                    print(f"RHB Batting Avg: {format_stat(processed_data.get('rhb_batting_avg'))}")
                    print(f"LHB OPS: {format_stat(processed_data.get('lhb_ops'))}")
                    print(f"RHB OPS: {format_stat(processed_data.get('rhb_ops'))}")
                except Exception as e:
                    print("Error displaying batter statistics:", str(e))
                
                # Add betting analysis
                print("\nBetting Analysis:")
                home_ml = game.get('home_team_moneyline')
                away_ml = game.get('away_team_moneyline')
                over_under = game.get('over_under_total')
                
                print(f"Current Moneyline: {home_ml if home_ml is not None else 'N/A'} / {away_ml if away_ml is not None else 'N/A'}")
                print(f"Current Over/Under: {over_under if over_under is not None else 'N/A'}")
                
                # Moneyline Analysis
                home_edge = get_betting_edge(pred['home_win_probability'], home_ml) if home_ml else None
                away_edge = get_betting_edge(1 - pred['home_win_probability'], away_ml) if away_ml else None
                
                # Only recommend bets with significant edge and high confidence
                if pred['prediction_confidence'] >= 0.6:
                    if home_edge and home_edge > 0.1:  # 10% edge threshold
                        print(f"Moneyline Recommendation: BET {game['home_team']} ({home_ml})")
                        print(f"Edge: {home_edge:.1%}")
                    elif away_edge and away_edge > 0.1:
                        print(f"Moneyline Recommendation: BET {game['away_team']} ({away_ml})")
                        print(f"Edge: {away_edge:.1%}")
                    else:
                        print("Moneyline Recommendation: Don't bet")
                else:
                    print("Moneyline Recommendation: Don't bet (low confidence)")
                
                # Over/Under Analysis
                if over_under and pred['prediction_confidence'] >= 0.6:
                    margin = abs(pred['total_runs'] - over_under)
                    if margin >= 1.5:  # Require at least 1.5 runs difference
                        if pred['total_runs'] > over_under:
                            print(f"Over/Under Recommendation: BET OVER {over_under}")
                            print(f"Margin: +{margin:.1f} runs")
                        else:
                            print(f"Over/Under Recommendation: BET UNDER {over_under}")
                            print(f"Margin: -{margin:.1f} runs")
                    else:
                        print("Over/Under Recommendation: Don't bet (margin too small)")
                else:
                    print("Over/Under Recommendation: Don't bet (low confidence or no line)")
        
        # Save predictions
        if predictions:
            try:
                save_predictions(predictions, 'game_predictions.csv')
                print("\nPredictions saved successfully to game_predictions.csv")
            except Exception as e:
                print(f"Error saving predictions: {e}")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 