import pandas as pd
import numpy as np
from datetime import datetime
import requests
from typing import Dict, List, Optional
import json
import os

class BettingAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
        self.performance_history = self.load_performance_history()
        
    def load_performance_history(self) -> Dict:
        """Load historical prediction performance"""
        if os.path.exists('prediction_performance.json'):
            with open('prediction_performance.json', 'r') as f:
                return json.load(f)
        return {
            'winner_predictions': {'correct': 0, 'total': 0},
            'moneyline_bets': {'wins': 0, 'losses': 0, 'total': 0},
            'over_under_bets': {'wins': 0, 'losses': 0, 'total': 0},
            'hit_predictions': {'correct': 0, 'total': 0}
        }
        
    def save_performance_history(self):
        """Save prediction performance history"""
        with open('prediction_performance.json', 'w') as f:
            json.dump(self.performance_history, f)
            
    def clean_team_name(self, team_name: str) -> str:
        """Remove city names from team names for matching"""
        # Common city prefixes to remove
        city_prefixes = ['New York', 'Boston', 'Chicago', 'Los Angeles', 'San Francisco', 
                        'Houston', 'Philadelphia', 'Washington', 'Miami', 'Seattle',
                        'Minnesota', 'Milwaukee', 'Cincinnati', 'Pittsburgh', 'Detroit',
                        'Cleveland', 'Kansas City', 'Baltimore', 'Tampa Bay', 'Texas',
                        'Arizona', 'Colorado', 'San Diego', 'Oakland', 'Toronto']
        
        for prefix in city_prefixes:
            if team_name.startswith(prefix):
                return team_name[len(prefix):].strip()
        return team_name
        
    def get_game_odds(self, date: str, home_team: str, away_team: str) -> Dict:
        """Fetch current odds for a specific game"""
        endpoint = f"{self.base_url}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,totals',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            games = response.json()
            
            # Clean team names for matching
            clean_home = self.clean_team_name(home_team)
            clean_away = self.clean_team_name(away_team)
            target_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%dT')
            
            # Find matching game
            for game in games:
                game_date = game['commence_time'].split('T')[0]
                game_home = self.clean_team_name(game['home_team'])
                game_away = self.clean_team_name(game['away_team'])
                
                if (game_date == target_date and 
                    game_home == clean_home and 
                    game_away == clean_away):
                    return game
                    
            return None
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return None
            
    def calculate_value_bets(self, 
                           predicted_prob: float, 
                           market_odds: float,
                           kelly_fraction: float = 0.5) -> Dict:
        """Calculate value bets using Kelly Criterion"""
        # Convert American odds to decimal
        if market_odds > 0:
            decimal_odds = (market_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(market_odds)) + 1
            
        # Calculate implied probability from market odds
        implied_prob = 1 / decimal_odds
        
        # Calculate edge
        edge = predicted_prob - implied_prob
        
        # Calculate Kelly Criterion
        if edge > 0:
            kelly = (edge * decimal_odds - (1 - predicted_prob)) / decimal_odds
            kelly = max(0, min(kelly * kelly_fraction, 1))  # Apply fraction and cap at 1
        else:
            kelly = 0
            
        return {
            'predicted_probability': predicted_prob,
            'implied_probability': implied_prob,
            'edge': edge,
            'kelly_fraction': kelly,
            'decimal_odds': decimal_odds,
            'market_odds': market_odds
        }
        
    def analyze_game(self, 
                    game_data: Dict, 
                    prediction_data: Dict) -> Dict:
        """Analyze a game for betting value"""
        # Get current odds
        odds_data = self.get_game_odds(
            game_data['date'],
            game_data['home_team'],
            game_data['away_team']
        )
        
        if not odds_data:
            return None
            
        # Extract relevant odds
        home_odds = None
        away_odds = None
        total_odds = None
        over_under = None
        
        # Find moneyline odds
        for bookmaker in odds_data['bookmakers']:
            if bookmaker['key'] == 'fanduel':  # Using FanDuel as primary source
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game_data['home_team']:
                                home_odds = outcome['price']
                            elif outcome['name'] == game_data['away_team']:
                                away_odds = outcome['price']
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                total_odds = outcome['price']
                                over_under = float(outcome['description'].split(' ')[1])
                break
        
        if not all([home_odds, away_odds, total_odds, over_under]):
            return None
            
        # Calculate value bets
        home_value = self.calculate_value_bets(
            float(prediction_data['home_win_probability'].strip('%')) / 100,
            home_odds
        )
        
        away_value = self.calculate_value_bets(
            1 - float(prediction_data['home_win_probability'].strip('%')) / 100,
            away_odds
        )
        
        # Calculate over/under value
        predicted_total = prediction_data['predicted_total']
        over_prob = 0.5 + (predicted_total - over_under) / 10  # Simple probability calculation
        over_value = self.calculate_value_bets(over_prob, total_odds)
        
        return {
            'game_id': odds_data['id'],
            'home_team': game_data['home_team'],
            'away_team': game_data['away_team'],
            'date': game_data['date'],
            'predicted_score': f"{prediction_data['predicted_away_runs']}-{prediction_data['predicted_home_runs']}",
            'home_value_bet': home_value,
            'away_value_bet': away_value,
            'over_under_value': over_value,
            'market_odds': {
                'home': home_odds,
                'away': away_odds,
                'total': total_odds,
                'over_under_line': over_under
            }
        }
        
    def get_betting_recommendations(self, 
                                  analysis: Dict, 
                                  min_edge: float = 0.05,
                                  min_kelly: float = 0.1) -> List[Dict]:
        """Generate betting recommendations based on analysis"""
        recommendations = []
        
        # Check moneyline value
        if analysis['home_value_bet']['edge'] > min_edge and analysis['home_value_bet']['kelly_fraction'] > min_kelly:
            recommendations.append({
                'type': 'moneyline',
                'team': analysis['home_team'],
                'odds': analysis['market_odds']['home'],
                'edge': analysis['home_value_bet']['edge'],
                'kelly': analysis['home_value_bet']['kelly_fraction'],
                'confidence': 'high' if analysis['home_value_bet']['edge'] > 0.1 else 'medium'
            })
            
        if analysis['away_value_bet']['edge'] > min_edge and analysis['away_value_bet']['kelly_fraction'] > min_kelly:
            recommendations.append({
                'type': 'moneyline',
                'team': analysis['away_team'],
                'odds': analysis['market_odds']['away'],
                'edge': analysis['away_value_bet']['edge'],
                'kelly': analysis['away_value_bet']['kelly_fraction'],
                'confidence': 'high' if analysis['away_value_bet']['edge'] > 0.1 else 'medium'
            })
            
        # Check over/under value
        if analysis['over_under_value']['edge'] > min_edge and analysis['over_under_value']['kelly_fraction'] > min_kelly:
            recommendations.append({
                'type': 'over_under',
                'line': analysis['market_odds']['over_under_line'],
                'odds': analysis['market_odds']['total'],
                'edge': analysis['over_under_value']['edge'],
                'kelly': analysis['over_under_value']['kelly_fraction'],
                'confidence': 'high' if analysis['over_under_value']['edge'] > 0.1 else 'medium'
            })
            
        return recommendations
        
    def update_performance(self, prediction_type: str, was_correct: bool):
        """Update prediction performance history"""
        if prediction_type == 'winner':
            self.performance_history['winner_predictions']['total'] += 1
            if was_correct:
                self.performance_history['winner_predictions']['correct'] += 1
        elif prediction_type == 'moneyline':
            self.performance_history['moneyline_bets']['total'] += 1
            if was_correct:
                self.performance_history['moneyline_bets']['wins'] += 1
            else:
                self.performance_history['moneyline_bets']['losses'] += 1
        elif prediction_type == 'over_under':
            self.performance_history['over_under_bets']['total'] += 1
            if was_correct:
                self.performance_history['over_under_bets']['wins'] += 1
            else:
                self.performance_history['over_under_bets']['losses'] += 1
        elif prediction_type == 'hits':
            self.performance_history['hit_predictions']['total'] += 1
            if was_correct:
                self.performance_history['hit_predictions']['correct'] += 1
                
        self.save_performance_history()
        
    def get_performance_summary(self) -> Dict:
        """Get summary of prediction performance"""
        summary = {}
        
        # Winner predictions
        winner = self.performance_history['winner_predictions']
        if winner['total'] > 0:
            summary['winner_accuracy'] = winner['correct'] / winner['total']
            
        # Moneyline bets
        moneyline = self.performance_history['moneyline_bets']
        if moneyline['total'] > 0:
            summary['moneyline_win_rate'] = moneyline['wins'] / moneyline['total']
            summary['moneyline_record'] = f"{moneyline['wins']}-{moneyline['losses']}"
            
        # Over/under bets
        over_under = self.performance_history['over_under_bets']
        if over_under['total'] > 0:
            summary['over_under_win_rate'] = over_under['wins'] / over_under['total']
            summary['over_under_record'] = f"{over_under['wins']}-{over_under['losses']}"
            
        # Hit predictions
        hits = self.performance_history['hit_predictions']
        if hits['total'] > 0:
            summary['hit_prediction_accuracy'] = hits['correct'] / hits['total']
            
        return summary

def main():
    # Example usage
    api_key = "df8571eddb753cdaa066497b9058722f"
    analyzer = BettingAnalyzer(api_key)
    
    # Example game data
    game_data = {
        'date': '2025-04-08',
        'home_team': 'Yankees',
        'away_team': 'Red Sox'
    }
    
    # Example prediction data
    prediction_data = {
        'home_win_probability': '65%',
        'predicted_home_runs': 4.5,
        'predicted_away_runs': 3.2
    }
    
    # Analyze game
    analysis = analyzer.analyze_game(game_data, prediction_data)
    if analysis:
        recommendations = analyzer.get_betting_recommendations(analysis)
        
        print("\nBetting Analysis:")
        print(f"{analysis['away_team']} @ {analysis['home_team']}")
        print(f"Predicted Score: {analysis['predicted_score']}")
        print("\nValue Bets:")
        for rec in recommendations:
            print(f"\n{rec['type'].upper()} - {rec['team'] if 'team' in rec else f'Line: {rec['line']}'}")
            print(f"Odds: {rec['odds']}")
            print(f"Edge: {rec['edge']:.1%}")
            print(f"Kelly: {rec['kelly']:.1%}")
            print(f"Confidence: {rec['confidence']}")
            
        # Print performance summary
        print("\nPrediction Performance:")
        summary = analyzer.get_performance_summary()
        for metric, value in summary.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.1%}")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 