from make_predictions import BaseballPredictions
import pandas as pd
import os
import numpy as np
from datetime import datetime

def validate_player(player_id: str, player_name: str, player_stats_df: pd.DataFrame) -> bool:
    """Validate if a player exists in our historical data."""
    try:
        # Convert player_id to string for comparison
        player_stats_df['player_id'] = player_stats_df['player_id'].astype(str)
        player_id = str(player_id)
        
        # Check if player exists
        player_exists = player_id in player_stats_df['player_id'].values
        if not player_exists:
            print(f"DEBUG: Player {player_name} (ID: {player_id}) not found in historical data")
        return player_exists
    except Exception as e:
        print(f"DEBUG: Error validating player {player_name}: {str(e)}")
        return False

def print_lineup_info(game_data: dict, player_stats_df: pd.DataFrame):
    """Print detailed lineup information with validation."""
    print("\nLineup Information:")
    print("-" * 80)
    
    # Home Team
    print(f"\n{game_data['home_team']} Lineup:")
    print(f"Pitcher: {game_data['home_pitcher']['name']} (ID: {game_data['home_pitcher']['player_id']})")
    if not validate_player(game_data['home_pitcher']['player_id'], game_data['home_pitcher']['name'], player_stats_df):
        print(f"⚠️  WARNING: No historical data found for pitcher {game_data['home_pitcher']['name']}")
    
    print("\nBatting Order:")
    for batter in game_data['home_lineup']:
        valid = validate_player(batter['player_id'], batter['name'], player_stats_df)
        warning = "⚠️  No historical data" if not valid else "✓"
        print(f"{batter['order']}. {batter['name']} ({batter['position']}) - ID: {batter['player_id']} - {warning}")
    
    # Away Team
    print(f"\n{game_data['away_team']} Lineup:")
    print(f"Pitcher: {game_data['away_pitcher']['name']} (ID: {game_data['away_pitcher']['player_id']})")
    if not validate_player(game_data['away_pitcher']['player_id'], game_data['away_pitcher']['name'], player_stats_df):
        print(f"⚠️  WARNING: No historical data found for pitcher {game_data['away_pitcher']['name']}")
    
    print("\nBatting Order:")
    for batter in game_data['away_lineup']:
        valid = validate_player(batter['player_id'], batter['name'], player_stats_df)
        warning = "⚠️  No historical data" if not valid else "✓"
        print(f"{batter['order']}. {batter['name']} ({batter['position']}) - ID: {batter['player_id']} - {warning}")

def get_pitcher_stats(pitcher_id: str, player_stats_df: pd.DataFrame) -> tuple:
    """Get pitcher's recent ERA from historical data."""
    try:
        # Convert player_id to string for comparison
        player_stats_df['player_id'] = player_stats_df['player_id'].astype(str)
        pitcher_id = str(pitcher_id)
        
        # Filter for pitcher and ensure is_pitcher is True
        pitcher_stats = player_stats_df[
            (player_stats_df['player_id'] == pitcher_id) & 
            (player_stats_df['is_pitcher'] == True)
        ].sort_values('date', ascending=False)
        
        if pitcher_stats.empty:
            print(f"DEBUG: No pitching stats found for ID: {pitcher_id}")
            return None
            
        recent_stats = pitcher_stats.iloc[0]
        era = recent_stats['era'] if pd.notna(recent_stats['era']) else None
        
        if era is not None:
            print(f"DEBUG: Found ERA {era:.2f} for pitcher ID: {pitcher_id}")
        else:
            print(f"DEBUG: ERA is null for pitcher ID: {pitcher_id}")
            
        return era
    except Exception as e:
        print(f"DEBUG: Error getting pitcher stats: {str(e)}")
        return None

def format_era(era: float) -> str:
    """Format ERA value for display."""
    if era is None:
        return 'N/A'
    return f'{era:.2f}'

def print_debug_info(processed_data, features_used):
    """Print detailed debug information about the prediction process"""
    print("\n=== DEBUG: Score Prediction Features ===")
    
    print("\n1. Weather & Environment Factors:")
    print(f"Temperature (normalized): {processed_data.get('temp_normalized', 'N/A')}")
    print(f"Wind Speed (normalized): {processed_data.get('wind_speed_normalized', 'N/A')}")
    print(f"Venue Park Factor: {processed_data.get('venue_park_factor', 'N/A')}")
    print(f"Is Roof Closed: {processed_data.get('is_roof_closed', 'N/A')}")
    
    print("\n2. Team Performance Metrics:")
    print(f"Home Team Last 10 Wins: {processed_data.get('home_team_last_10_wins', 'N/A')}")
    print(f"Away Team Last 10 Wins: {processed_data.get('away_team_last_10_wins', 'N/A')}")
    print(f"Home Team Weighted Runs: {processed_data.get('home_team_weighted_runs_scored', 'N/A')}")
    print(f"Away Team Weighted Runs: {processed_data.get('away_team_weighted_runs_scored', 'N/A')}")
    print(f"Home Team Runs StdDev: {processed_data.get('home_team_runs_std', 'N/A')}")
    print(f"Away Team Runs StdDev: {processed_data.get('away_team_runs_std', 'N/A')}")
    
    print("\n3. Pitcher Statistics:")
    print(f"LHP ERA: {processed_data.get('lhp_era', 'N/A')}")
    print(f"RHP ERA: {processed_data.get('rhp_era', 'N/A')}")
    print(f"LHP WHIP: {processed_data.get('lhp_whip', 'N/A')}")
    print(f"RHP WHIP: {processed_data.get('rhp_whip', 'N/A')}")
    print(f"LHP K/9: {processed_data.get('lhp_k_per_9', 'N/A')}")
    print(f"RHP K/9: {processed_data.get('rhp_k_per_9', 'N/A')}")
    
    print("\n4. Batter Statistics:")
    print(f"LHB Batting Avg: {processed_data.get('lhb_batting_avg', 'N/A')}")
    print(f"RHB Batting Avg: {processed_data.get('rhb_batting_avg', 'N/A')}")
    print(f"LHB OPS: {processed_data.get('lhb_ops', 'N/A')}")
    print(f"RHB OPS: {processed_data.get('rhb_ops', 'N/A')}")
    
    print("\n5. Venue Statistics:")
    print(f"Venue Avg Total Runs: {processed_data.get('venue_avg_total_runs', 'N/A')}")
    print(f"Venue StdDev Total Runs: {processed_data.get('venue_std_total_runs', 'N/A')}")
    print(f"Venue Park Factor: {processed_data.get('venue_park_factor', 'N/A')}")
    
    print("\n6. Matchup History:")
    print(f"Days Since Last Matchup: {processed_data.get('days_since_last_matchup', 'N/A')}")
    print(f"Matchup Trend: {processed_data.get('matchup_trend', 'N/A')}")
    print(f"Weighted Avg Runs in Matchup: {processed_data.get('weighted_avg_runs_in_matchup', 'N/A')}")
    print(f"Matchup Runs StdDev: {processed_data.get('matchup_runs_std', 'N/A')}")
    
    print("\n7. Features Used in Model:")
    print("Features included in prediction:")
    for feature in features_used:
        print(f"- {feature}")

def main():
    # Initialize the predictor with betting API key
    print("Loading models...")
    predictor = BaseballPredictions(betting_api_key="df8571eddb753cdaa066497b9058722f")
    
    # Load player stats
    print("\nLoading player statistics...")
    try:
        player_stats_df = pd.read_csv('history/player_stats.csv')
        print(f"Loaded statistics for {len(player_stats_df['player_id'].unique())} players")
    except Exception as e:
        print(f"Error loading player statistics: {e}")
        return
    
    # Load upcoming games from CSV
    print("\nLoading upcoming games...")
    try:
        games_df = pd.read_csv('upcoming_games_template.csv')
        print(f"Loaded {len(games_df)} games")
        
        # Process each game
        predictions = []
        for _, game in games_df.iterrows():
            print(f"\n{'='*80}")
            print(f"Processing game: {game['away_team']} @ {game['home_team']} on {game['date']}")
            
            # Format game data
            game_data = {
                'game_id': str(game['game_id']),
                'date': game['date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'venue_id': str(game['venue_id']),
                'temp': float(game['temp']),
                'wind_speed': float(game['wind_speed']),
                'condition': game['condition'],
                'home_team_moneyline': float(game['home_team_moneyline']),
                'away_team_moneyline': float(game['away_team_moneyline']),
                'over_under_total': float(game['over_under_total']),
                'home_pitcher_player_id': str(game['home_pitcher_player_id']),
                'away_pitcher_player_id': str(game['away_pitcher_player_id']),
                'home_pitcher': {
                    'player_id': str(game['home_pitcher_player_id']),
                    'name': game['home_pitcher_name']
                },
                'away_pitcher': {
                    'player_id': str(game['away_pitcher_player_id']),
                    'name': game['away_pitcher_name']
                },
                'home_lineup': [],
                'away_lineup': []
            }
            
            # Add home lineup
            for i in range(1, 10):
                batter_data = {
                    'player_id': str(game[f'home_batter{i}_player_id']),
                    'name': str(game[f'home_batter{i}_name']),
                    'position': str(game[f'home_batter{i}_position']),
                    'order': i
                }
                game_data['home_lineup'].append(batter_data)
                game_data[f'home_batter{i}_player_id'] = str(game[f'home_batter{i}_player_id'])
            
            # Add away lineup
            for i in range(1, 10):
                batter_data = {
                    'player_id': str(game[f'away_batter{i}_player_id']),
                    'name': str(game[f'away_batter{i}_name']),
                    'position': str(game[f'away_batter{i}_position']),
                    'order': i
                }
                game_data['away_lineup'].append(batter_data)
                game_data[f'away_batter{i}_player_id'] = str(game[f'away_batter{i}_player_id'])
            
            # Print and validate lineup information
            print_lineup_info(game_data, player_stats_df)
            
            # Make prediction
            print("\nGenerating prediction...")
            print("DEBUG: Processing game data for prediction...")
            
            # Get processed data and features for debugging
            print("DEBUG: Getting processed data...")
            processed_data = predictor.data_processor.process_game(pd.DataFrame([game_data]))
            print("DEBUG: Getting score features...")
            score_features = predictor.prepare_prediction_data(game_data, 'home_runs').columns.tolist()
            
            print("DEBUG: Making prediction...")
            pred = predictor.predict_game(game_data)
            
            # Print debug information
            print_debug_info(processed_data.iloc[0], score_features)
            
            print("DEBUG: Getting pitcher stats...")
            # Get pitcher ERAs
            home_pitcher_era = get_pitcher_stats(game_data['home_pitcher']['player_id'], player_stats_df)
            away_pitcher_era = get_pitcher_stats(game_data['away_pitcher']['player_id'], player_stats_df)
            
            # Get betting analysis
            if predictor.betting_analyzer:
                betting_analysis = predictor.betting_analyzer.analyze_game(game_data, pred)
                if betting_analysis:
                    recommendations = predictor.betting_analyzer.get_betting_recommendations(
                        betting_analysis,
                        min_edge=0.1,  # Only recommend bets with >10% edge
                        min_kelly=0.15  # Only recommend bets with >15% Kelly criterion
                    )
                    pred['betting_analysis'] = betting_analysis
                    pred['betting_recommendations'] = recommendations
            
            predictions.append(pred)
            
            # Print prediction
            if 'error' in pred:
                print(f"Error: {pred['error']}")
            else:
                print(f"\nPrediction Results:")
                print(f"Starting Pitchers:")
                print(f"{game_data['home_team']}: {game_data['home_pitcher']['name']} (ERA: {format_era(home_pitcher_era)})")
                print(f"{game_data['away_team']}: {game_data['away_pitcher']['name']} (ERA: {format_era(away_pitcher_era)})")
                print(f"\nHome Win Probability: {pred['home_win_probability']}%")
                print(f"Predicted Score: {pred['away_team']} {pred['predicted_away_runs']} - {pred['predicted_home_runs']} {pred['home_team']}")
                print(f"Total Runs: {pred['predicted_total']}")
                print(f"Confidence: {pred['confidence']}%")
                
                # Print betting recommendations if available
                if 'betting_recommendations' in pred and pred['betting_recommendations']:
                    print("\nHigh Confidence Betting Recommendations:")
                    for rec in pred['betting_recommendations']:
                        if rec['confidence'] == 'high':
                            if rec['type'] == 'moneyline':
                                print(f"🎯 Bet on {rec['team']} ({rec['odds']}) - {rec['edge']*100:.1f}% edge")
                            else:
                                print(f"🎯 Bet {rec['type'].upper()} {rec['line']} ({rec['odds']}) - {rec['edge']*100:.1f}% edge")
                            print(f"   Recommended stake: {rec['kelly']*100:.1f}% of betting unit")
                else:
                    print("\nNo high-confidence betting opportunities found")
        
        # Save predictions to CSV
        print("\nSaving predictions to predictions.csv...")
        pd.DataFrame(predictions).to_csv('predictions.csv', index=False)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 