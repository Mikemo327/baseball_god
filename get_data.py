from datetime import datetime, timedelta
import statsapi
import pandas as pd
import os
from tqdm import tqdm

def get_games_by_date(date_str):
    """Get games for a specific date using statsapi."""
    try:
        # Validate date format
        datetime.strptime(date_str, '%Y-%m-%d')
        schedule = statsapi.schedule(date=date_str)
        
        # Filter for completed games only
        completed_games = [game for game in schedule if game.get('status') == 'Final']
        return {'dates': [{'games': completed_games}]}
    except ValueError as e:
        print(f"Invalid date format. Please use YYYY-MM-DD format: {e}")
        return None
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return None

def get_game_boxscore(game_id):
    """Get boxscore for a specific game using statsapi."""
    try:
        game_data = statsapi.get('game', {'gamePk': game_id})
        if not game_data or 'liveData' not in game_data:
            print(f"No live data available for game {game_id}")
            return None
        return game_data['liveData']['boxscore']
    except Exception as e:
        print(f"Error fetching boxscore for game {game_id}: {e}")
        return None

def calculate_woba(stats):
    """Calculate wOBA using standard weights."""
    bb = float(stats.get('baseOnBalls', 0))
    hbp = float(stats.get('hitByPitch', 0))
    singles = float(stats.get('hits', 0)) - float(stats.get('doubles', 0)) - float(stats.get('triples', 0)) - float(stats.get('homeRuns', 0))
    doubles = float(stats.get('doubles', 0))
    triples = float(stats.get('triples', 0))
    hr = float(stats.get('homeRuns', 0))
    ab = float(stats.get('atBats', 0))
    sf = float(stats.get('sacFlies', 0))
    
    if (ab + bb + sf + hbp) == 0:
        return 'N/A'
    
    woba = (0.69 * bb + 0.72 * hbp + 0.89 * singles + 1.27 * doubles + 1.62 * triples + 2.10 * hr) / (ab + bb + sf + hbp)
    return round(woba, 3)

def calculate_fip(stats):
    """Calculate FIP using standard weights."""
    ip = float(stats.get('inningsPitched', 0))
    hr = float(stats.get('homeRuns', 0))
    bb = float(stats.get('baseOnBalls', 0))
    k = float(stats.get('strikeOuts', 0))
    
    if ip == 0:
        return 'N/A'
    
    fip_constant = 3.10  # League average FIP constant
    fip = ((13 * hr + 3 * bb - 2 * k) / ip) + fip_constant
    return round(fip, 2)

def safe_float(value, default=0.0, min_val=None, max_val=None):
    """
    Safely convert value to float with optional range validation.
    Returns default if conversion fails or value is outside valid range.
    """
    try:
        result = float(value) if value is not None else default
        if min_val is not None and result < min_val:
            print(f"Warning: Value {result} is below minimum {min_val}, using default")
            return default
        if max_val is not None and result > max_val:
            print(f"Warning: Value {result} is above maximum {max_val}, using default")
            return default
        return result
    except (ValueError, TypeError):
        return default

def get_player_details(player_info):
    """Extract player-level statistics with improved validation."""
    try:
        if not player_info:
            print("Warning: Empty player info received")
            return None

        person = player_info.get('person', {})
        position = player_info.get('position', {})
        season_stats = player_info.get('seasonStats', {})
        
        # Get basic player info with defaults
        player_details = {
            'player_id': person.get('id', 'N/A'),
            'name': person.get('fullName', 'N/A'),
            'position': position.get('abbreviation', 'N/A'),
            'team_id': player_info.get('parentTeamId', 'N/A'),
            'bat_side': person.get('batSide', {}).get('code', 'R'),
            'throw_arm': person.get('pitchHand', {}).get('code', 'R'),
            'is_batter': False,
            'is_pitcher': False
        }

        # Process batting stats if available
        if 'batting' in season_stats and season_stats['batting'].get('gamesPlayed', 0) > 0:
            batting_stats = season_stats['batting']
            player_details.update({
                'is_batter': True,
                'batting_avg': safe_float(batting_stats.get('avg'), default='.000'),
                'obp': safe_float(batting_stats.get('obp'), default='.000'),
                'slg': safe_float(batting_stats.get('slg'), default='.000'),
                'ops': safe_float(batting_stats.get('ops'), default='.000'),
                'hits': safe_float(batting_stats.get('hits', 0), min_val=0),
                'home_runs': safe_float(batting_stats.get('homeRuns', 0), min_val=0),
                'rbis': safe_float(batting_stats.get('rbi', 0), min_val=0),
                'runs': safe_float(batting_stats.get('runs', 0), min_val=0),
                'doubles': safe_float(batting_stats.get('doubles', 0), min_val=0),
                'triples': safe_float(batting_stats.get('triples', 0), min_val=0),
                'stolen_bases': safe_float(batting_stats.get('stolenBases', 0), min_val=0),
                'caught_stealing': safe_float(batting_stats.get('caughtStealing', 0), min_val=0),
                'babip': safe_float(batting_stats.get('babip'), default='.000')
            })

        # Process pitching stats if available
        if 'pitching' in season_stats and season_stats['pitching'].get('gamesPlayed', 0) > 0:
            pitching_stats = season_stats['pitching']
            player_details.update({
                'is_pitcher': True,
                'position': 'P' if position.get('abbreviation') == 'P' else player_details['position'],
                'era': safe_float(pitching_stats.get('era'), default='0.00'),
                'whip': safe_float(pitching_stats.get('whip'), default='0.00'),
                'innings_pitched': safe_float(pitching_stats.get('inningsPitched', 0), min_val=0),
                'strikeouts': safe_float(pitching_stats.get('strikeOuts', 0), min_val=0),
                'walks': safe_float(pitching_stats.get('baseOnBalls', 0), min_val=0),
                'hits_allowed': safe_float(pitching_stats.get('hits', 0), min_val=0),
                'runs_allowed': safe_float(pitching_stats.get('runs', 0), min_val=0),
                'earned_runs': safe_float(pitching_stats.get('earnedRuns', 0), min_val=0),
                'home_runs_allowed': safe_float(pitching_stats.get('homeRuns', 0), min_val=0),
                'wins': safe_float(pitching_stats.get('wins', 0), min_val=0),
                'losses': safe_float(pitching_stats.get('losses', 0), min_val=0),
                'saves': safe_float(pitching_stats.get('saves', 0), min_val=0),
                'k_per_9': safe_float(pitching_stats.get('strikeoutsPer9Inn'), default='0.00'),
                'bb_per_9': safe_float(pitching_stats.get('walksPer9Inn'), default='0.00'),
                'hr_per_9': safe_float(pitching_stats.get('homeRunsPer9'), default='0.00'),
                'strike_pct': safe_float(pitching_stats.get('strikePercentage'), default='.000'),
                'ground_outs_to_airouts': safe_float(pitching_stats.get('groundOutsToAirouts'), min_val=0)
            })

        # Process fielding stats if available
        if 'fielding' in season_stats:
            fielding_stats = season_stats['fielding']
            player_details.update({
                'errors': safe_float(fielding_stats.get('errors', 0), min_val=0),
                'assists': safe_float(fielding_stats.get('assists', 0), min_val=0),
                'putouts': safe_float(fielding_stats.get('putouts', 0), min_val=0)
            })

        # Calculate advanced metrics if possible
        if player_details['is_batter']:
            woba = calculate_woba(season_stats.get('batting', {}))
            player_details['woba'] = woba if isinstance(woba, float) and 0 <= woba <= 1 else 'N/A'

        if player_details['is_pitcher']:
            fip = calculate_fip(season_stats.get('pitching', {}))
            player_details['fip'] = fip if isinstance(fip, float) and fip >= 0 else 'N/A'

        return player_details
    except Exception as e:
        print(f"Error processing player details: {str(e)}")
        return None

def get_team_stats(team_data):
    """Extract team-level statistics."""
    team_stats = team_data.get('teamStats', {})
    batting_stats = team_stats.get('batting', {})
    pitching_stats = team_stats.get('pitching', {})
    fielding_stats = team_stats.get('fielding', {})
    
    # Calculate team wOBA
    team_woba = calculate_woba(batting_stats)
    team_fip = calculate_fip(pitching_stats)
    
    return {
        # Team batting stats
        'team_batting_avg': batting_stats.get('avg', 'N/A'),
        'team_obp': batting_stats.get('obp', 'N/A'),
        'team_slg': batting_stats.get('slg', 'N/A'),
        'team_ops': batting_stats.get('ops', 'N/A'),
        'team_runs': batting_stats.get('runs', 'N/A'),
        'team_home_runs': batting_stats.get('homeRuns', 'N/A'),
        'team_hits': batting_stats.get('hits', 'N/A'),
        'team_doubles': batting_stats.get('doubles', 'N/A'),
        'team_triples': batting_stats.get('triples', 'N/A'),
        'team_stolen_bases': batting_stats.get('stolenBases', 'N/A'),
        'team_woba': team_woba,
        'team_babip': batting_stats.get('babip', 'N/A'),
        # Team pitching stats
        'team_era': pitching_stats.get('era', 'N/A'),
        'team_whip': pitching_stats.get('whip', 'N/A'),
        'team_strikeouts': pitching_stats.get('strikeOuts', 'N/A'),
        'team_walks': pitching_stats.get('baseOnBalls', 'N/A'),
        'team_hits_allowed': pitching_stats.get('hits', 'N/A'),
        'team_runs_allowed': pitching_stats.get('runs', 'N/A'),
        'team_home_runs_allowed': pitching_stats.get('homeRuns', 'N/A'),
        'team_fip': team_fip,
        'team_k_per_9': pitching_stats.get('strikeoutsPer9Inn', 'N/A'),
        'team_bb_per_9': pitching_stats.get('walksPer9Inn', 'N/A'),
        'team_hr_per_9': pitching_stats.get('homeRunsPer9', 'N/A'),
        # Team fielding stats
        'team_fielding_pct': fielding_stats.get('fieldingPct', 'N/A'),
        'team_errors': fielding_stats.get('errors', 'N/A')
    }

def get_team_info(game):
    """Get team and venue information from game data."""
    # Extract team names without city names
    home_team_full = game.get('home_name', 'Unknown')
    away_team_full = game.get('away_name', 'Unknown')
    
    # Extract just the team name (without city) using the get_team_name function
    home_team = get_team_name(home_team_full)
    away_team = get_team_name(away_team_full)
    
    # Get game scores
    home_runs = game.get('home_score', 0)
    away_runs = game.get('away_score', 0)
    
    # Determine winner (1 for home team win, 0 for away team win)
    home_win = None  # Will be None for ties or incomplete games
    if home_runs > away_runs:
        home_win = 1  # Home team won
    elif away_runs > home_runs:
        home_win = 0  # Away team won
    
    venue_name = game.get('venue_name', 'Unknown')
    venue_id = game.get('venue_id', 'Unknown')
    date = game.get('game_date', 'Unknown')
    
    return {
        'date': date,
        'game_id': game.get('game_id', 'Unknown'),
        'home_team': home_team,
        'away_team': away_team,
        'venue_name': venue_name,
        'venue_id': venue_id,
        'home_runs': home_runs,
        'away_runs': away_runs,
        'home_win': home_win
    }

def get_team_name(team_name):
    """Extract just the team name without city, handling special cases."""
    if team_name == 'Unknown':
        return 'Unknown'
    
    # Special cases for teams with two-word names
    special_cases = {
        'White Sox': 'White Sox',
        'Red Sox': 'Red Sox',
        'Blue Jays': 'Blue Jays',
        'Devil Rays': 'Devil Rays',
        'Rays': 'Rays'
    }
    
    # Check if the team name contains any of the special cases
    for special_name in special_cases:
        if special_name in team_name:
            return special_cases[special_name]
    
    # For all other teams, take the last word
    return team_name.split()[-1]

def get_player_stats(game_id, game_date):
    """Get player statistics for a specific game."""
    try:
        boxscore = get_game_boxscore(game_id)
        if not boxscore:
            return None

        player_stats = []
        teams = boxscore.get('teams', {})
        
        for team_type in ['home', 'away']:
            team_data = teams.get(team_type, {})
            team_info = get_team_info({
                'game_id': game_id,
                'game_date': game_date,
                'home_name': team_data.get('team', {}).get('name', 'Unknown'),
                'away_name': team_data.get('team', {}).get('name', 'Unknown'),
                'home_score': team_data.get('teamStats', {}).get('batting', {}).get('runs', 0),
                'away_score': team_data.get('teamStats', {}).get('batting', {}).get('runs', 0),
                'venue_name': team_data.get('team', {}).get('venue', {}).get('name', 'Unknown'),
                'venue_id': team_data.get('team', {}).get('venue', {}).get('id', 'Unknown')
            })
            
            players = team_data.get('players', {})
            for player_id, player_info in players.items():
                if player_info.get('person', {}).get('id'):
                    player_details = get_player_details(player_info)
                    if player_details:
                        player_details.update({
                            'date': game_date,
                            'game_id': game_id,
                            'team': team_info['home_team'] if team_type == 'home' else team_info['away_team'],
                            'team_type': team_type
                        })
                        player_stats.append(player_details)
        
        return player_stats
    except Exception as e:
        print(f"Error getting player stats for game {game_id}: {e}")
        return None

def get_venue_info(game):
    """Get venue and weather information from game data."""
    try:
        # Get the full game data to access venue information
        game_data = statsapi.get('game', {'gamePk': game.get('game_id')})
        if not game_data or 'gameData' not in game_data:
            return None
            
        venue = game_data['gameData'].get('venue', {})
        weather = game_data['gameData'].get('weather', {})
        
        venue_info = {
            'venue_id': venue.get('id', ''),
            'venue_name': venue.get('name', ''),
            'venue_city': venue.get('location', {}).get('city', ''),
            'venue_state': venue.get('location', {}).get('state', ''),
            'venue_country': venue.get('location', {}).get('country', ''),
            'venue_timezone': venue.get('timeZone', {}).get('id', ''),
            'venue_latitude': venue.get('location', {}).get('defaultCoordinates', {}).get('latitude', ''),
            'venue_longitude': venue.get('location', {}).get('defaultCoordinates', {}).get('longitude', ''),
            'venue_altitude': venue.get('location', {}).get('altitude', ''),
            # Add weather information
            'temp': weather.get('temp', ''),
            'wind_speed': weather.get('wind', '').split('mph')[0].strip() if weather.get('wind', '').endswith('mph') else '',
            'wind_direction': ' '.join(weather.get('wind', '').split()[:-1]) if weather.get('wind') else '',
            'condition': weather.get('condition', '')
        }
        
        return venue_info
    except Exception as e:
        print(f"Error getting venue and weather info: {e}")
        return None

def main():
    # Create history directory if it doesn't exist
    os.makedirs('history', exist_ok=True)
    
    # Set date range for data collection
    # start_date = '2025-04-06'  # Opening Day
    # end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2024-09-01'  # Opening Day
    end_date = '2025-04-07'

    
    print("Starting data collection...")
    
    # Calculate total days for progress bar
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1
    
    all_games = []
    all_player_stats = []
    all_venues = {}  # Dictionary to store unique venues
    current_date = start
    
    # Create progress bar for dates
    with tqdm(total=total_days, desc="Fetching games", unit="day") as pbar:
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            games_data = get_games_by_date(date_str)
            
            if games_data and 'dates' in games_data and games_data['dates']:
                for date_info in games_data['dates']:
                    if 'games' in date_info:
                        for game in date_info['games']:
                            if game.get('status') == 'Final':
                                game_id = game.get('game_id')
                                
                                # Get venue and weather information
                                venue_info = get_venue_info(game)
                                venue_id = venue_info['venue_id'] if venue_info else ''
                                
                                # Get team names without city names
                                home_team = get_team_name(game['home_name'])
                                away_team = get_team_name(game['away_name'])
                                
                                # Combine game and venue data
                                game_data = {
                                    'game_id': game_id,
                                    'date': date_str,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_runs': game.get('home_score', 0),
                                    'away_runs': game.get('away_score', 0),
                                    'home_win': 1 if game.get('home_score', 0) > game.get('away_score', 0) else 0
                                }
                                
                                # Add venue and weather information if available
                                if venue_info:
                                    game_data.update({
                                        'venue_id': venue_info.get('venue_id', ''),
                                        'venue_name': venue_info.get('venue_name', ''),
                                        'venue_city': venue_info.get('venue_city', ''),
                                        'venue_state': venue_info.get('venue_state', ''),
                                        'venue_country': venue_info.get('venue_country', ''),
                                        'venue_timezone': venue_info.get('venue_timezone', ''),
                                        'venue_latitude': venue_info.get('venue_latitude', ''),
                                        'venue_longitude': venue_info.get('venue_longitude', ''),
                                        'venue_altitude': venue_info.get('venue_altitude', ''),
                                        'temp': venue_info.get('temp', ''),
                                        'wind_speed': venue_info.get('wind_speed', ''),
                                        'wind_direction': venue_info.get('wind_direction', ''),
                                        'condition': venue_info.get('condition', '')
                                    })
                                
                                all_games.append(game_data)
                                
                                # Get player stats for this game
                                game_player_stats = get_player_stats(game_id, date_str)
                                if game_player_stats:
                                    all_player_stats.extend(game_player_stats)
            
            current_date += timedelta(days=1)
            pbar.update(1)
    
    # Convert to DataFrames
    games_df = pd.DataFrame(all_games)
    player_stats_df = pd.DataFrame(all_player_stats)
    
    print(f"\nFound {len(games_df)} games")
    print(f"Found {len(player_stats_df)} player appearances")
    
    # Save games data (now includes venue info)
    games_file = 'history/games.csv'
    if os.path.exists(games_file):
        existing_games_df = pd.read_csv(games_file)
        combined_games_df = pd.concat([existing_games_df, games_df]).drop_duplicates(subset=['game_id'], keep='last')
        # Reorder columns to put date first
        cols = combined_games_df.columns.tolist()
        cols.remove('date')
        cols.insert(0, 'date')
        combined_games_df = combined_games_df[cols]
        combined_games_df.to_csv(games_file, index=False)
        print(f"Updated games file with {len(combined_games_df)} total games")
    else:
        # Reorder columns to put date first
        cols = games_df.columns.tolist()
        cols.remove('date')
        cols.insert(0, 'date')
        games_df = games_df[cols]
        games_df.to_csv(games_file, index=False)
        print(f"Created new games file with {len(games_df)} games")
    
    # Save player stats data
    player_stats_file = 'history/player_stats.csv'
    if os.path.exists(player_stats_file):
        existing_player_stats_df = pd.read_csv(player_stats_file)
        combined_player_stats_df = pd.concat([existing_player_stats_df, player_stats_df]).drop_duplicates(
            subset=['game_id', 'player_id'], keep='last'
        )
        # Reorder columns to put date first
        cols = combined_player_stats_df.columns.tolist()
        cols.remove('date')
        cols.insert(0, 'date')
        combined_player_stats_df = combined_player_stats_df[cols]
        combined_player_stats_df.to_csv(player_stats_file, index=False)
        print(f"Updated player stats file with {len(combined_player_stats_df)} total player appearances")
    else:
        # Reorder columns to put date first
        cols = player_stats_df.columns.tolist()
        cols.remove('date')
        cols.insert(0, 'date')
        player_stats_df = player_stats_df[cols]
        player_stats_df.to_csv(player_stats_file, index=False)
        print(f"Created new player stats file with {len(player_stats_df)} player appearances")
    
    print("\nData collection completed successfully!")

if __name__ == "__main__":
    main() 