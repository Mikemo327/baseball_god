import pandas as pd

# Load the processed data
df = pd.read_csv('processed_data.csv')
player_stats = pd.read_csv('history/player_stats.csv')

# Find games where Ohtani played
ohtani_games = player_stats[player_stats['name'].str.contains('Ohtani', case=False, na=False)]
if len(ohtani_games) > 0:
    print("\nFound Ohtani in these games:")
    for game_id in ohtani_games['game_id'].unique():
        game_data = df[df['game_id'] == game_id].iloc[0]
        print(f"\nGame {game_id}: {game_data['away_team']} @ {game_data['home_team']} on {game_data['date']}")
        
        # Get Ohtani's team for this game
        ohtani_stats = ohtani_games[ohtani_games['game_id'] == game_id].iloc[0]
        ohtani_team = ohtani_stats['team']
        
        # Get opposing pitchers (pitchers not on Ohtani's team)
        game_pitchers = player_stats[
            (player_stats['game_id'] == game_id) & 
            (player_stats['position'] == 'P') &
            (player_stats['team'] != ohtani_team)
        ][['name', 'throw_arm', 'era', 'whip', 'k_per_9']]
        
        print(f"\nOpposing Pitchers (not on {ohtani_team}):")
        print(game_pitchers.to_string())
        
        print("\nOhtani's stats for this game:")
        print(f"Bats: {ohtani_stats['bat_side']}")
        print(f"AVG: {ohtani_stats['batting_avg']:.3f}")
        print(f"OPS: {ohtani_stats['ops']:.3f}")
        print(f"SLG: {ohtani_stats['slg']:.3f}")
        
        print("\nGame Context:")
        print(f"Temperature: {game_data['temp']}°F")
        print(f"Weather: {game_data['condition_x']}")
        print(f"Venue: {game_data['venue_name']}")
        print(f"Park Factor: {game_data['venue_park_factor']:.2f}")
else:
    print("No games found with Ohtani in the dataset.")

# Let's look at a specific game and examine the pitcher-batter matchups
def examine_matchup(game_id, pitcher_name, batter_name):
    # Load the original player stats for this game
    player_stats = pd.read_csv('history/player_stats.csv')
    
    # Get the game data
    game_data = df[df['game_id'] == game_id].iloc[0]
    
    # Get pitcher and batter data for this game
    pitcher_data = player_stats[(player_stats['game_id'] == game_id) & 
                              (player_stats['name'] == pitcher_name)].iloc[0]
    batter_data = player_stats[(player_stats['game_id'] == game_id) & 
                              (player_stats['name'] == batter_name)].iloc[0]
    
    # Check if pitcher and batter are on the same team
    if pitcher_data['team'] == batter_data['team']:
        print(f"\nERROR: {pitcher_name} and {batter_name} are on the same team ({pitcher_data['team']})!")
        print("They would not face each other in this game.")
        return
    
    print(f"\nMatchup Analysis for Game {game_id}")
    print(f"{game_data['away_team']} @ {game_data['home_team']}")
    print(f"Date: {game_data['date']}")
    print(f"\nPitcher: {pitcher_name} ({pitcher_data['team']})")
    print(f"Throws: {pitcher_data['throw_arm']}")
    print(f"ERA: {pitcher_data['era']:.2f}")
    print(f"WHIP: {pitcher_data['whip']:.2f}")
    print(f"K/9: {pitcher_data['k_per_9']:.2f}")
    
    print(f"\nBatter: {batter_name} ({batter_data['team']})")
    print(f"Bats: {batter_data['bat_side']}")
    print(f"AVG: {batter_data['batting_avg']:.3f}")
    print(f"OPS: {batter_data['ops']:.3f}")
    print(f"SLG: {batter_data['slg']:.3f}")
    
    # Get handedness matchup stats from processed data
    if pitcher_data['throw_arm'] == 'L':
        print(f"\nLHP Stats for this game:")
        print(f"LHP ERA: {game_data['lhp_era']:.2f}")
        print(f"LHP WHIP: {game_data['lhp_whip']:.2f}")
        print(f"LHP K/9: {game_data['lhp_k_per_9']:.2f}")
    else:
        print(f"\nRHP Stats for this game:")
        print(f"RHP ERA: {game_data['rhp_era']:.2f}")
        print(f"RHP WHIP: {game_data['rhp_whip']:.2f}")
        print(f"RHP K/9: {game_data['rhp_k_per_9']:.2f}")
    
    if batter_data['bat_side'] == 'L':
        print(f"\nLHB Stats for this game:")
        print(f"LHB AVG: {game_data['lhb_batting_avg']:.3f}")
        print(f"LHB OPS: {game_data['lhb_ops']:.3f}")
        print(f"LHB SLG: {game_data['lhb_slg']:.3f}")
    else:
        print(f"\nRHB Stats for this game:")
        print(f"RHB AVG: {game_data['rhb_batting_avg']:.3f}")
        print(f"RHB OPS: {game_data['rhb_ops']:.3f}")
        print(f"RHB SLG: {game_data['rhb_slg']:.3f}")

# Let's examine a specific game matchup
# First, let's find a game and its players
print("Available games and players:")
player_stats = pd.read_csv('history/player_stats.csv')
sample_game = player_stats['game_id'].iloc[0]
print("\nPlayers in game", sample_game)
print("\nPitchers:")
print(player_stats[(player_stats['game_id'] == sample_game) & 
                  (player_stats['position'] == 'P')][['name', 'throw_arm']].to_string())
print("\nBatters:")
print(player_stats[(player_stats['game_id'] == sample_game) & 
                  (player_stats['position'] != 'P')][['name', 'bat_side']].head().to_string())

# Example: examine_matchup(sample_game, "Pitcher Name", "Batter Name")
# Uncomment and fill in actual names from the printed list

# Let's examine MacKenzie Gore vs CJ Abrams
print("\nExamining specific matchup:")
examine_matchup(778429, "MacKenzie Gore", "CJ Abrams")

# Let's examine Ohtani vs Glasnow specifically
print("\nDetailed Ohtani vs Glasnow matchup:")
examine_matchup(778418, "Tyler Glasnow", "Shohei Ohtani")

# Let's examine Ohtani vs a proper opposing pitcher
print("\nExamining Ohtani vs Wheeler matchup:")
examine_matchup(778434, "Zack Wheeler", "Shohei Ohtani") 