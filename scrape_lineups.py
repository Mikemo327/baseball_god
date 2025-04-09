import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pytz
import re
import os

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_player_id(href):
    """Extract player ID from href attribute."""
    if not href:
        return None
    match = re.search(r'/player/[^/]+-(\d+)', href)
    return match.group(1) if match else None

def print_game_header(game_num, total_games, game_id, away_team, home_team, away_pitcher_info, home_pitcher_info):
    """Print a formatted game header with pitcher information."""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}Game {game_num}/{total_games} (ID: {game_id}){Colors.ENDC}")
    print(f"{Colors.BLUE}{away_team}{Colors.ENDC} @ {Colors.GREEN}{home_team}{Colors.ENDC}")
    
    # Print pitcher information
    print(f"{Colors.BOLD}Starting Pitchers:{Colors.ENDC}")
    if away_pitcher_info:
        print(f"{Colors.BLUE}Away: {away_pitcher_info['name']} (ID: {away_pitcher_info['player_id']}) - {away_pitcher_info['hand']}, ERA: {away_pitcher_info['era']}, SO: {away_pitcher_info['so']}{Colors.ENDC}")
    if home_pitcher_info:
        print(f"{Colors.GREEN}Home: {home_pitcher_info['name']} (ID: {home_pitcher_info['player_id']}) - {home_pitcher_info['hand']}, ERA: {home_pitcher_info['era']}, SO: {home_pitcher_info['so']}{Colors.ENDC}")
    
    print(f"{Colors.BOLD}{'-'*80}{Colors.ENDC}")

def print_player_info(player_name, player_id, position, is_home):
    """Print formatted player information."""
    team_color = Colors.GREEN if is_home else Colors.BLUE
    print(f"{team_color}{position:<4}{Colors.ENDC} {player_name:<25} (ID: {player_id})")

def extract_pitcher_info(pitcher_element):
    """Extract pitcher information from a pitcher summary element."""
    if not pitcher_element:
        return None
    
    # Find the pitcher name element
    pitcher_name_div = pitcher_element.find('div', class_='starting-lineups__pitcher-name')
    if not pitcher_name_div:
        return None
    
    # Get the name from the link inside the name div
    pitcher_link = pitcher_name_div.find('a', class_='starting-lineups__pitcher--link')
    if not pitcher_link:
        return None
    
    # Extract name and ID
    name = pitcher_link.text.strip()
    player_id = extract_player_id(pitcher_link.get('href'))
    
    # Get pitching stats
    hand = pitcher_element.find('span', class_='starting-lineups__pitcher-pitch-hand')
    era = pitcher_element.find('span', class_='starting-lineups__pitcher-era')
    so = pitcher_element.find('span', class_='starting-lineups__pitcher-strikeouts')
    
    # Clean up the strikeouts and ERA values by removing 'SO' and 'ERA'
    so_value = so.text.strip().replace(' SO', '') if so else 'N/A'
    era_value = era.text.strip().replace(' ERA', '') if era else 'N/A'
    
    return {
        'name': name,
        'player_id': player_id,
        'hand': hand.text.strip() if hand else 'N/A',
        'era': era_value,
        'so': so_value
    }

def scrape_lineups(date_str):
    """Scrape lineups from MLB.com for a specific date."""
    url = f'https://www.mlb.com/starting-lineups/{date_str}'
    try:
        print(f"\n{Colors.BOLD}Fetching lineups from: {url}{Colors.ENDC}")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        lineups = []
        matchups = soup.find_all('div', class_='starting-lineups__matchup')
        
        if not matchups:
            print(f"{Colors.RED}No matchup containers found for {date_str}. Checking page structure...{Colors.ENDC}")
            print("First 500 characters of response:")
            print(response.text[:500])
            return None
        
        print(f"\n{Colors.BOLD}Found {len(matchups)} matchups{Colors.ENDC}")
        total_players = 0
        
        for i, matchup in enumerate(matchups, 1):
            game_id = matchup.get('data-gamepk')
            if not game_id:
                print(f"{Colors.RED}Warning: No game ID found for matchup {i}{Colors.ENDC}")
                continue
            
            # Get team names
            away_team_name = matchup.find('span', class_='starting-lineups__team-name--away')
            if away_team_name:
                away_team_name = away_team_name.find('a', class_='starting-lineups__team-name--link')
                if away_team_name:
                    away_team_name = away_team_name.text.strip()
            
            home_team_name = matchup.find('span', class_='starting-lineups__team-name--home')
            if home_team_name:
                home_team_name = home_team_name.find('a', class_='starting-lineups__team-name--link')
                if home_team_name:
                    home_team_name = home_team_name.text.strip()
            
            # Get pitcher information - find all pitcher summaries
            # IMPORTANT: The first pitcher summary is ALWAYS the away pitcher, the second is ALWAYS the home pitcher
            # There may be an empty pitcher summary between them
            pitcher_summaries = matchup.find_all('div', class_='starting-lineups__pitcher-summary')
            
            # Initialize pitcher info variables
            away_pitcher_info = None
            home_pitcher_info = None
            
            # Find the away pitcher (first non-empty pitcher summary)
            for summary in pitcher_summaries:
                if summary.find('div', class_='starting-lineups__pitcher-name'):
                    away_pitcher_info = extract_pitcher_info(summary)
                    break
            
            # Find the home pitcher (second non-empty pitcher summary)
            found_away = False
            for summary in pitcher_summaries:
                if summary.find('div', class_='starting-lineups__pitcher-name'):
                    if found_away:
                        home_pitcher_info = extract_pitcher_info(summary)
                        break
                    found_away = True
            
            # Print game header with pitcher information
            print_game_header(i, len(matchups), game_id, away_team_name, home_team_name, away_pitcher_info, home_pitcher_info)
            
            # Add pitcher information to lineups list
            if away_pitcher_info:
                lineups.append({
                    'game_id': game_id,
                    'team': away_team_name,
                    'is_home': False,
                    'position': 'P',
                    'player_id': away_pitcher_info['player_id'],
                    'player_name': away_pitcher_info['name'],
                    'pitcher_hand': away_pitcher_info['hand'],
                    'pitcher_era': away_pitcher_info['era'],
                    'pitcher_so': away_pitcher_info['so']
                })
                total_players += 1
            
            if home_pitcher_info:
                lineups.append({
                    'game_id': game_id,
                    'team': home_team_name,
                    'is_home': True,
                    'position': 'P',
                    'player_id': home_pitcher_info['player_id'],
                    'player_name': home_pitcher_info['name'],
                    'pitcher_hand': home_pitcher_info['hand'],
                    'pitcher_era': home_pitcher_info['era'],
                    'pitcher_so': home_pitcher_info['so']
                })
                total_players += 1
            
            # Process away team
            away_lineup = matchup.find('ol', class_='starting-lineups__team--away')
            if away_lineup:
                away_players = away_lineup.find_all('a', class_='starting-lineups__player--link')
                if away_players:
                    print(f"\n{Colors.BLUE}Away Team Lineup:{Colors.ENDC}")
                    for player in away_players:
                        player_name = player.text.strip()
                        player_href = player.get('href', '')
                        player_id = extract_player_id(player_href)
                        position = player.find_next_sibling('span', class_='starting-lineups__player--position')
                        
                        if player_id and position:
                            print_player_info(player_name, player_id, position.text.strip(), False)
                            lineups.append({
                                'game_id': game_id,
                                'team': away_team_name,
                                'is_home': False,
                                'position': position.text.strip(),
                                'player_id': player_id,
                                'player_name': player_name,
                                'pitcher_hand': None,
                                'pitcher_era': None,
                                'pitcher_so': None
                            })
                            total_players += 1
            
            # Process home team
            home_lineup = matchup.find('ol', class_='starting-lineups__team--home')
            if home_lineup:
                home_players = home_lineup.find_all('a', class_='starting-lineups__player--link')
                if home_players:
                    print(f"\n{Colors.GREEN}Home Team Lineup:{Colors.ENDC}")
                    for player in home_players:
                        player_name = player.text.strip()
                        player_href = player.get('href', '')
                        player_id = extract_player_id(player_href)
                        position = player.find_next_sibling('span', class_='starting-lineups__player--position')
                        
                        if player_id and position:
                            print_player_info(player_name, player_id, position.text.strip(), True)
                            lineups.append({
                                'game_id': game_id,
                                'team': home_team_name,
                                'is_home': True,
                                'position': position.text.strip(),
                                'player_id': player_id,
                                'player_name': player_name,
                                'pitcher_hand': None,
                                'pitcher_era': None,
                                'pitcher_so': None
                            })
                            total_players += 1
        
        if total_players > 0:
            print(f"\n{Colors.BOLD}Found {total_players} total players in lineups{Colors.ENDC}")
            df = pd.DataFrame(lineups)
            
            if not os.path.exists('mlb_data'):
                os.makedirs('mlb_data')
                
            output_file = f'mlb_data/lineups_{date_str}.csv'
            df.to_csv(output_file, index=False)
            print(f"\n{Colors.GREEN}Saved lineups to {output_file}{Colors.ENDC}")
            return df
        else:
            print(f"\n{Colors.RED}No lineups found for {date_str}. They may not be posted yet.{Colors.ENDC}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Error fetching lineups: {str(e)}{Colors.ENDC}")
        return None
    except Exception as e:
        print(f"{Colors.RED}Error processing lineups: {str(e)}{Colors.ENDC}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    # Set timezone to EST
    est = pytz.timezone('US/Eastern')
    current_date = datetime.now(est)
    date_str = current_date.strftime('%Y-%m-%d')
    
    # Scrape and save lineups
    lineups_df = scrape_lineups(date_str)
    if lineups_df is not None:
        print("\nSuccessfully scraped lineups!")
        print("\nSample of lineups:")
        print(lineups_df.head())
    else:
        print("\nNo lineups were found or there was an error.")

if __name__ == "__main__":
    main() 