
import json

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

        with open('championFull.json') as f:
            self.champion_id_to_name = json.load(f)

        self.champion_names_in_game = self.get_champion_names()
        self.champions_to_roles = self.convert_champions_to_roles(self.champion_names_in_game)

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

        return transformed_data
    

    def convert_champions_to_roles(self, match_data):
        
        transformed_data = []
        for player_data in match_data:

            champion_id = player_data[1]
            champion_name = self.champion_id_to_name.get(champion_id, 'Unknown')
            role = self.get_role(champion_name)
            new_player_data = player_data.copy()
            new_player_data.append(role)
            transformed_data.append(new_player_data)

        return transformed_data