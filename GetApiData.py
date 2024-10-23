
import time
import requests
from urllib.parse import quote
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PlayerInfo:
    summoner_name: str
    champion_id: int
    team_id: int
    primary_rune: str
    secondary_rune: str

    
@dataclass
class GetApiData():

    def __init__(self, player_name, tag, api_key = 'dd'):
        self.api_key = api_key
        self.player_name = player_name
        self.tag = tag


    def get_puuid_by_riot_id(self):
  
        # Using europe region for EUNE accounts
        base_url = "https://europe.api.riotgames.com"
        
        # Encode both the game name and tagline separately
        encoded_name = quote(self.player_name)
        encoded_tag = quote(self.tag_line)
        
        endpoint = f"{base_url}/riot/account/v1/accounts/by-riot-id/{encoded_name}/{encoded_tag}"
        
        headers = {
            "X-Riot-Token": self.api_key
        }
        
        print(f"Making request to: {endpoint}")
        
        response = requests.get(endpoint, headers=headers)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return response['puuid']
    
    def get_summoner_by_puuid(self, puuid: str, region: str = "eun1"):
        """First get summoner data using PUUID"""
        base_url = f"https://{region}.api.riotgames.com"
        endpoint = f"{base_url}/lol/summoner/v4/summoners/by-puuid/{puuid}"
        
        headers = {
            "X-Riot-Token": self.api_key
        }

        response = requests.get(endpoint, headers=headers)
        print(f"Summoner lookup status: {response.status_code}")
        return response.json()
    

    def get_active_match_details(self, puuid: str, region: str = self.tag) -> List[PlayerInfo]:
        """Get champion and rune information for all players in an active match"""
        base_url = f"https://{region}.api.riotgames.com"
        endpoint = f"{base_url}/lol/spectator/v5/active-games/by-summoner/{puuid}"
        
        try:
            headers = {
            "X-Riot-Token": self.api_key
            }
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            match_data = response.json()
            
            players_info = []
            
            for participant in match_data['participants']:

                primary_style = participant['perks']['perkIds'][0]
                secondary_style = participant['perks']['perkIds'][4]

                
                player_info = PlayerInfo(
                    summoner_name=participant['puuid'],
                    champion_id=participant['championId'],
                    team_id=participant['teamId'],
                    primary_rune=primary_style or 'Unknown',
                    secondary_rune=secondary_style or 'Unknown'
                )
                
                players_info.append(player_info)
            
            return players_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching match data: {e}")
            return []