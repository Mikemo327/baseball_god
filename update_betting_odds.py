import pandas as pd
import requests
from datetime import datetime
import sys

def standardize_team_name(team_name):
    """Standardize team names to match between CSV and API"""
    # Map of various team names to standard names
    team_map = {
        'Yankees': 'New York Yankees',
        'Tigers': 'Detroit Tigers',
        'Red Sox': 'Boston Red Sox',
        'Blue Jays': 'Toronto Blue Jays',
        'Rays': 'Tampa Bay Rays',
        'Orioles': 'Baltimore Orioles',
        'Guardians': 'Cleveland Guardians',
        'White Sox': 'Chicago White Sox',
        'Twins': 'Minnesota Twins',
        'Royals': 'Kansas City Royals',
        'Astros': 'Houston Astros',
        'Angels': 'Los Angeles Angels',
        'Athletics': 'Oakland Athletics',
        'Mariners': 'Seattle Mariners',
        'Rangers': 'Texas Rangers',
        'Mets': 'New York Mets',
        'Braves': 'Atlanta Braves',
        'Phillies': 'Philadelphia Phillies',
        'Marlins': 'Miami Marlins',
        'Nationals': 'Washington Nationals',
        'Cardinals': 'St. Louis Cardinals',
        'Cubs': 'Chicago Cubs',
        'Brewers': 'Milwaukee Brewers',
        'Pirates': 'Pittsburgh Pirates',
        'Reds': 'Cincinnati Reds',
        'Dodgers': 'Los Angeles Dodgers',
        'Giants': 'San Francisco Giants',
        'Padres': 'San Diego Padres',
        'Rockies': 'Colorado Rockies',
        'Diamondbacks': 'Arizona Diamondbacks'
    }
    
    # First try direct mapping
    if team_name in team_map:
        return team_map[team_name]
    
    # If not found, try reverse lookup (full name to short name)
    for short_name, full_name in team_map.items():
        if team_name == full_name:
            return full_name
    
    return team_name

def get_current_odds(api_key):
    """Fetch current MLB odds from the API"""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h,totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None

def print_available_games(odds_data):
    """Print all available games from the API"""
    print("\nAvailable games from API:")
    print("=" * 80)
    for game in odds_data:
        game_date = datetime.strptime(game['commence_time'].split('T')[0], '%Y-%m-%d')
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get odds
        home_ml = None
        away_ml = None
        total = None
        
        for bookmaker in game['bookmakers']:
            if bookmaker['key'] == 'fanduel':
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                home_ml = outcome['price']
                            elif outcome['name'] == away_team:
                                away_ml = outcome['price']
                    elif market['key'] == 'totals':
                        total = float(market['outcomes'][0]['point'])
                break
        
        print(f"\nDate: {game_date.date()}")
        print(f"Game: {away_team} @ {home_team}")
        print(f"Home ML: {home_ml if home_ml else 'N/A'}")
        print(f"Away ML: {away_ml if away_ml else 'N/A'}")
        print(f"O/U: {total if total else 'N/A'}")
        print("-" * 40)

def update_odds_in_csv(csv_path, api_key):
    """Update the odds in the CSV file with current odds"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} games from CSV")
        
        # Get current odds
        odds_data = get_current_odds(api_key)
        if not odds_data:
            print("Failed to fetch odds data")
            return
        
        print(f"Fetched odds for {len(odds_data)} games")
        
        # Print available games
        print_available_games(odds_data)
        
        print("\nGames in CSV:")
        for _, row in df.iterrows():
            print(f"{row['away_team']} @ {row['home_team']} on {row['date']}")
        
        # Track matches
        matches_found = 0
        
        # Process each game in the CSV
        for idx, row in df.iterrows():
            game_date = datetime.strptime(row['date'], '%Y-%m-%d')
            csv_home_team = standardize_team_name(row['home_team'])
            csv_away_team = standardize_team_name(row['away_team'])
            
            print(f"\nLooking for match: {csv_away_team} @ {csv_home_team} on {game_date.date()}")
            
            # Look for matching game in odds data
            for game in odds_data:
                odds_date = datetime.strptime(game['commence_time'].split('T')[0], '%Y-%m-%d')
                odds_home = game['home_team']
                odds_away = game['away_team']
                
                # Check if this is the matching game (check both home/away combinations)
                is_match = (
                    game_date.date() == odds_date.date() and 
                    ((csv_home_team == odds_home and csv_away_team == odds_away) or
                     (csv_home_team == odds_away and csv_away_team == odds_home))
                )
                
                if is_match:
                    matches_found += 1
                    print(f"Found match: {odds_away} @ {odds_home} on {odds_date.date()}")
                    
                    # Extract odds
                    home_ml = None
                    away_ml = None
                    total = None
                    
                    for bookmaker in game['bookmakers']:
                        if bookmaker['key'] == 'fanduel':  # Using FanDuel as primary source
                            for market in bookmaker['markets']:
                                if market['key'] == 'h2h':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == odds_home:
                                            # If teams are swapped, swap the odds
                                            if csv_home_team == odds_away:
                                                away_ml = outcome['price']
                                            else:
                                                home_ml = outcome['price']
                                        elif outcome['name'] == odds_away:
                                            # If teams are swapped, swap the odds
                                            if csv_home_team == odds_away:
                                                home_ml = outcome['price']
                                            else:
                                                away_ml = outcome['price']
                                elif market['key'] == 'totals':
                                    total = float(market['outcomes'][0]['point'])
                            break
                    
                    # Update the DataFrame if odds were found
                    if home_ml is not None:
                        print(f"Updating home moneyline: {home_ml}")
                        df.at[idx, 'home_team_moneyline'] = home_ml
                    if away_ml is not None:
                        print(f"Updating away moneyline: {away_ml}")
                        df.at[idx, 'away_team_moneyline'] = away_ml
                    if total is not None:
                        print(f"Updating over/under: {total}")
                        df.at[idx, 'over_under_total'] = total
                    
                    break
        
        print(f"\nFound and updated {matches_found} games")
        
        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved updated odds to {csv_path}")
        
    except Exception as e:
        print(f"Error updating odds: {e}")
        return

if __name__ == "__main__":
    API_KEY = "df8571eddb753cdaa066497b9058722f"
    CSV_PATH = "upcoming_games_template.csv"
    
    print("Updating betting odds...")
    update_odds_in_csv(CSV_PATH, API_KEY) 