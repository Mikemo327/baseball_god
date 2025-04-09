import pandas as pd
import os
import json
from datetime import datetime

def create_upcoming_games_csv():
    """Create a template for upcoming games CSV file with mandatory 1 pitcher and 9 batters per team"""
    # Define the columns
    columns = [
        'game_id', 'date', 'home_team', 'away_team', 'venue_id', 'temp', 'wind_speed', 'condition',
        'home_pitcher_player_id', 'home_pitcher_name', 'away_pitcher_player_id', 'away_pitcher_name'
    ]
    
    # Add columns for exactly 9 batters for each team
    for i in range(1, 10):
        columns.extend([
            f'home_batter{i}_player_id', f'home_batter{i}_name', f'home_batter{i}_order',
            f'away_batter{i}_player_id', f'away_batter{i}_name', f'away_batter{i}_order'
        ])
    
    # Create empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)
    
    # Add example row
    example = {
        'game_id': '20250408_NYY_BOS',
        'date': '2025-04-08',
        'home_team': 'Yankees',
        'away_team': 'Red Sox',
        'venue_id': '3313',
        'temp': 65,
        'wind_speed': 8,
        'condition': 'Partly Cloudy',
        'home_pitcher_player_id': 'COLE001',
        'home_pitcher_name': 'Gerrit Cole',
        'away_pitcher_player_id': 'SALE001',
        'away_pitcher_name': 'Chris Sale'
    }
    
    # Add example batters (9 for each team)
    home_batters = [
        {'name': 'Aaron Judge', 'order': 1, 'player_id': 'JUDGE001'},
        {'name': 'Gleyber Torres', 'order': 2, 'player_id': 'TORRE001'},
        {'name': 'Anthony Rizzo', 'order': 3, 'player_id': 'RIZZO001'},
        {'name': 'DJ LeMahieu', 'order': 4, 'player_id': 'LEMAH001'},
        {'name': 'Giancarlo Stanton', 'order': 5, 'player_id': 'STANT001'},
        {'name': 'Harrison Bader', 'order': 6, 'player_id': 'BADER001'},
        {'name': 'Oswaldo Cabrera', 'order': 7, 'player_id': 'CABRE001'},
        {'name': 'Jose Trevino', 'order': 8, 'player_id': 'TREVI001'},
        {'name': 'Isiah Kiner-Falefa', 'order': 9, 'player_id': 'KINER001'}
    ]
    
    away_batters = [
        {'name': 'Rafael Devers', 'order': 1, 'player_id': 'DEVER001'},
        {'name': 'Trevor Story', 'order': 2, 'player_id': 'STORY001'},
        {'name': 'Jarren Duran', 'order': 3, 'player_id': 'DURAN001'},
        {'name': 'Masataka Yoshida', 'order': 4, 'player_id': 'YOSHI001'},
        {'name': 'Justin Turner', 'order': 5, 'player_id': 'TURNE001'},
        {'name': 'Alex Verdugo', 'order': 6, 'player_id': 'VERDU001'},
        {'name': 'Triston Casas', 'order': 7, 'player_id': 'CASAS001'},
        {'name': 'Connor Wong', 'order': 8, 'player_id': 'WONG001'},
        {'name': 'Enmanuel Valdez', 'order': 9, 'player_id': 'VALDE001'}
    ]
    
    # Add batter information to example
    for i, batter in enumerate(home_batters, 1):
        example[f'home_batter{i}_player_id'] = batter['player_id']
        example[f'home_batter{i}_name'] = batter['name']
        example[f'home_batter{i}_order'] = batter['order']
    
    for i, batter in enumerate(away_batters, 1):
        example[f'away_batter{i}_player_id'] = batter['player_id']
        example[f'away_batter{i}_name'] = batter['name']
        example[f'away_batter{i}_order'] = batter['order']
    
    # Add example to DataFrame
    df = pd.concat([df, pd.DataFrame([example])], ignore_index=True)
    
    # Save to CSV
    df.to_csv('upcoming_games.csv', index=False)
    print("Created upcoming_games.csv template with example data")
    
    # Print instructions
    print("\nInstructions:")
    print("1. Open upcoming_games.csv in Excel or a text editor")
    print("2. Replace the example data with your actual game data")
    print("3. IMPORTANT: Each game MUST have:")
    print("   - A unique game_id (format: YYYYMMDD_HOME_AWAY)")
    print("   - Exactly 1 pitcher for each team with a unique player_id (home_pitcher_player_id, home_pitcher_name, away_pitcher_player_id, away_pitcher_name)")
    print("   - Exactly 9 batters for each team with unique player_ids (home_batter1_player_id through home_batter9_player_id, away_batter1_player_id through away_batter9_player_id)")
    print("4. Add more rows for additional games")
    print("5. Save the file")
    print("6. Run predict_upcoming_games.py to generate predictions")

def validate_upcoming_games(csv_path: str) -> bool:
    """Validate that the upcoming games CSV has the correct structure"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check for duplicate game_ids
        if df['game_id'].duplicated().any():
            print("Error: Duplicate game_ids found")
            return False
        
        # Track all player_ids to check for duplicates
        all_player_ids = set()
        
        for idx, row in df.iterrows():
            # Check game_id
            if pd.isna(row['game_id']):
                print(f"Error in row {idx + 1}: Missing game_id")
                return False
            
            # Check pitchers
            if pd.isna(row['home_pitcher_player_id']) or pd.isna(row['home_pitcher_name']):
                print(f"Error in row {idx + 1}: Missing home pitcher information")
                return False
            if pd.isna(row['away_pitcher_player_id']) or pd.isna(row['away_pitcher_name']):
                print(f"Error in row {idx + 1}: Missing away pitcher information")
                return False
            
            # Add pitcher player_ids to the set
            all_player_ids.add(row['home_pitcher_player_id'])
            all_player_ids.add(row['away_pitcher_player_id'])
            
            # Check batters (must have exactly 9 for each team)
            for team in ['home', 'away']:
                for i in range(1, 10):
                    if pd.isna(row[f'{team}_batter{i}_player_id']) or pd.isna(row[f'{team}_batter{i}_name']) or \
                       pd.isna(row[f'{team}_batter{i}_order']):
                        print(f"Error in row {idx + 1}: Missing {team} batter {i} information")
                        return False
                    
                    # Check for duplicate player_ids
                    player_id = row[f'{team}_batter{i}_player_id']
                    if player_id in all_player_ids:
                        print(f"Error in row {idx + 1}: Duplicate player_id {player_id} found")
                        return False
                    all_player_ids.add(player_id)
        
        print("Validation successful: All games have required game_id, pitchers and batters with unique player_ids")
        return True
        
    except Exception as e:
        print(f"Error validating upcoming games: {e}")
        return False

def main():
    create_upcoming_games_csv()
    if os.path.exists('upcoming_games.csv'):
        validate_upcoming_games('upcoming_games.csv')

if __name__ == "__main__":
    main() 