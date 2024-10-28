
import json
import pandas as pd
from roleidentification import pull_data, get_roles
from roleidentification.utilities import get_champion_roles, get_team_roles


class TransformDataFromApi():

    def __init__(self, data):
        self.data = data

        self.AD_Range_Shooter = ['Akshan','Smolder','Nilah','Urgot','Jayce','Corki','Kindred','Graves','Tristana', 'Jinx', 'Caitlyn', 'Ezreal', 'MissFortune', 'Varus', 'KogMaw', 'Samira','Lucian','Ashe', 'Aphelios', 'Kalista', 'Senna', 'Draven', 'Jhin', 'Vayne', 'Xayah', 'Sivir', 'Twitch', 'Kaisa', 'Quinn', 'Zeri']
        self.AD_Melee_Bruiser = ['Belveth','Gnar','LeeSin','Trundle','Fiora','Warwick','Riven','Gangplank','RekSai','Aatrox', 'Camille', 'Darius', 'Fiora', 'Garen', 'Hecarim', 'Illaoi', 'Irelia', 'JarvanIV', 'Jax', 'Kled', 'Nasus', 'Olaf', 'Pantheon', 'Renekton', 'Sett', 'Tryndamere', 'Vi', 'MonkeyKing', 'XinZhao', 'Yone','Yorick', 'Kayn']
        self.AD_Melee_Tank = ['DrMundo', 'KSante','Poppy','Sion','TahmKench','Shen']
        self.AD_Melee_Assassin = ['Khazix','Briar','Naafiri','MasterYi','Nocturne','Pyke','Qiyana','Rengar','Shaco','Talon','Viego','Yasuo','Zed']

        self.AP_Range_Mage = ['Ahri','Hwei','Aurora','Malzahar','Nidalee','Teemo','Kayle','Milio','Morgana','Nami','Vex','Zilean','Zoe','Zyra','Xerath','Yuumi','Ziggs','Vladimir','Neeko','Viktor','Soraka','Syndra','Veigar','Velkoz','Taliyah','TwistedFate','Orianna','Sona','Renata','Ryze','Lissandra','Janna','Seraphine','Lisandra','Lulu','Lux','Swain','Karma','Karthus','Ivern','Anivia','Annie','AurelionSol','Azir','Bard','Brand','Cassiopeia','Heimerdinger']
        self.AP_Melee_Assassin = ['Akali','Ekko','Evelynn','Fizz','Kassadin','Katarina']
        self.AP_Melee_Bruiser = ['Gwen','Sylas','Blitzcrank','Rakan','Diana','Gragas','Lillia','Rumble','Shyvana','Udyr','Volibear']
        self.AP_Range_Assassin = ['Leblanc','Elise','FiddleSticks','Kennen']
        self.AP_Melee_Tank = ['Malphite','Mordekaiser','Nunu','Ornn','Nautilus','Rell','Rammus','Braum','Maokai','Sejuani','Leona','Alistar','Singed','Skarner','Taric','Thresh','Zac','Amumu','Chogath','Galio']

        self.champions = {
            'AD_Range_Shooter': self.AD_Range_Shooter,
            'AD_Melee_Bruiser': self.AD_Melee_Bruiser,
            'AD_Melee_Tank': self.AD_Melee_Tank,
            'AD_Melee_Assassin':self.AD_Melee_Assassin,
            'AP_Range_Mage': self.AP_Range_Mage,
            'AP_Melee_Assassin': self.AP_Melee_Assassin,
            'AP_Melee_Bruiser': self.AP_Melee_Bruiser,
            'AP_Range_Assassin': self.AP_Range_Assassin,
            'AP_Melee_Tank': self.AP_Melee_Tank}

        with open('championFull.json', encoding='utf-8') as f:
            self.champion_id_to_name = json.load(f)

        self.champion_names_in_game = self.get_champion_names()
        self.champions_to_roles = self.convert_champions_to_roles(self.champion_names_in_game)

        self.final_columns = self.load_final_columns()
        self.player_positions = self.get_champion_position()

    def get_champion_names(self):

        
        id_to_name = {}
        for champ_name, champ_data in self.champion_id_to_name['data'].items():
            champ_id = int(champ_data['key'])  
            id_to_name[champ_id] = champ_name

        # transform the lists
        transformed_data = []
        for player_data in self.data:

            champion_id = player_data[1]
            champion_name = id_to_name.get(champion_id, 'Unknown')
            new_player_data = player_data.copy()
            new_player_data[1] = champion_name
            transformed_data.append(new_player_data)

        print(f"Champion names in game: {transformed_data}")
        return transformed_data
    

    def convert_champions_to_roles(self, match_data):
        transformed_data = []

        def get_role(champion_name):
            # Look through all roles in self.champions dictionary
            for role, champions in self.champions.items():
                if champion_name in champions:
                    return role
            return 'Unknown'  # Return Unknown if champion not found in any role list

        for player_data in match_data:
            # Champion name is already in player_data[1]
            champion_name = player_data[1]
            # Get role using our dictionary lookup
            role = get_role(champion_name)

            new_player_data = player_data.copy()
            new_player_data[1] = role
            transformed_data.append(new_player_data)

        return transformed_data
    

    def get_champion_position(self):
        try:
            print("Getting champion roles data...")
            champion_roles = pull_data()
            
            champion_ids = []
            
            # Just collect the champion IDs directly since they're already in position 1
            for player_data in self.data:
                champion_id = player_data[1]  # Champion ID is already at index 1
                champion_ids.append(champion_id)
            
            print(f"Champion IDs for position detection: {champion_ids}")
            
            # Make sure we have all players
            if len(champion_ids) == 10:
                # Important: get_team_roles expects two lists of 5 champions each
                team1_champs = champion_ids[:5]  # First 5 champions
                team2_champs = champion_ids[5:]  # Last 5 champions
                
                team1_roles = get_roles(champion_roles, team1_champs)
                team2_roles = get_roles(champion_roles, team2_champs)
                
                # Combine roles from both teams
                all_roles = team1_roles + team2_roles
                return all_roles
            else:
                print(f"Warning: Expected 10 champions, got {len(champion_ids)}")
                return ['UNKNOWN'] * len(champion_ids)
            
        except Exception as e:
            print(f"Error in get_champion_position: {str(e)}")
            print(f"Current data format: {self.data}")
            raise

    def load_final_columns(self):
        final_columns = pd.read_csv('processed_data\other_features_df.csv').columns.tolist()
        # add champion_id column as first coluimns
        final_columns.insert(0, 'champion_id')
   
        return final_columns
    
    def create_dataframe(self):
        
        df = pd.DataFrame(index=[0], columns=self.final_columns)
        df = df.fillna(0)
        # first of all - column with champion id of the player we are forecasting items for

        df.loc[0, 'champion_id'] = self.data[0][1]

        # second - columns with champion and position of all players in game

        for i in range(10):
            player_role = self.champions_to_roles[i][1]
            player_position = self.player_positions[i]

            team = 'Friendly' if self.data[i][2] == self.data[0][2] else 'Enemy'

            map_position_to_proper_name = {
            'TOP': 'Top',
            'JUNGLE': 'Jungler',
            'MIDDLE': 'Middle',
            'BOTTOM': 'Bottom',
            'UTILITY': 'Utility'
            }
            column_name = f'{team} {map_position_to_proper_name[player_position]}_player_role'

            df.loc[0, column_name]= 1

        # third - mastiers
        
            primary_rune = self.data[i][3]
            primary_col_name = f'Primary Rune_{primary_rune}'
            df.loc[0, primary_col_name] = 1

            secondary_rune = self.data[i][4]
            secondary_col_name = f'Secondary Rune_{secondary_rune}'
            df.loc[0, secondary_col_name] = 1

        return df

