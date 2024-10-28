import streamlit as st
import pandas as pd
from GetApiData import GetApiData  
from TransformDataFromApi import TransformDataFromApi  
from GetModelPredictions import GetModelPredictions  

class LoLPredictionApp:
    def __init__(self, items_df, other_features_df, item_encoder, champion_encoder):
        self.api_key = st.secrets["riot_api_key"]  # Store API key in Streamlit secrets
        self.setup_page()

        self.item_encoder = item_encoder
        self.champion_encoder = champion_encoder
        self.items_df = items_df
        self.other_features_df = other_features_df

    def setup_page(self):
        """Configure initial page layout"""
        st.set_page_config(
            page_title="LoL Item Predictor",
            page_icon="🎮",
            layout="wide"
        )
        st.title("League of Legends Item Predictor")
        st.markdown("Enter summoner details to get item recommendations")



    
    def get_user_input(self):
        REGIONS_BY_SERVER = {
            'Europe': ['EUW1', 'EUN1', 'TR1', 'RU'],
            'Americas': ['NA1', 'BR1', 'LA1', 'LA2'],
            'Asia': ['KR', 'JP1', 'OC1', 'PH2', 'SG2', 'TH2', 'TW2', 'VN2']
        }
        
        SERVER_TO_REGION = {
            region: server.lower()
            for server, regions in REGIONS_BY_SERVER.items()
            for region in regions
        }
        
        with st.form("summoner_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                summoner_name = st.text_input("Summoner Name")
            
            with col2:
                summoner_tag = st.text_input("Tag")

            with col3:
                region = st.selectbox(
                    "Region",
                    options=[
                        f"{region} ({server})"
                        for server, regions in REGIONS_BY_SERVER.items()
                        for region in regions
                    ],
                    format_func=lambda x: x.split(" (")[0],
                    help="Select your game region"
                )
                # Extract region code and get corresponding server
                region_code = region.split(" (")[0]
                server = SERVER_TO_REGION[region_code]
            
            submit_button = st.form_submit_button("Get Recommendations")

            return summoner_name, summoner_tag, server, region_code, submit_button
        
        
    def process_data(self, summoner_name, summoner_tag, server, region):
        """Process the input data through API and transformations"""
        try:
            # Get API data
            api_handler = GetApiData(summoner_name, summoner_tag, self.api_key, server)
            puuid = api_handler.get_puuid_by_riot_id()

            match_data = api_handler.get_active_match_details(puuid, region)
            print('before transformer')
            # Transform data
            transformer = TransformDataFromApi(match_data)

            print('data put into transformer')
            processed_data = transformer.create_dataframe()

            print(f'Processed data: {processed_data.head()}')
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None
        
    def make_predictions(self, processed_data):
        """Make predictions using the model"""
        try:
            predictor = GetModelPredictions(
                model_path="lol_itemization_model.pth",
                data=processed_data,
                item_encoder=self.item_encoder,
                champion_encoder=self.champion_encoder,
                items_df=self.items_df,
                other_features_df=self.other_features_df
            )
            
            predictions = predictor.predict(processed_data)
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return None
        

    def display_results(self, predictions):
        """Display the predictions in a nice format"""
        if predictions:
            st.success("Recommendations ready!")
            
            st.subheader("Recommended Items")
            for i, item in enumerate(predictions, 1):
                st.write(f"{i}. {item}")
            
            
        else:
            st.warning("Could not generate recommendations")


    def run(self):
        """Main application loop"""
        summoner_name, summoner_tag, server, region_code, submitted = self.get_user_input()

        
        if submitted:
            with st.spinner("Processing..."):
                if not summoner_name or not summoner_tag or not server:
                    st.error("Please enter both summoner name, tag and choose server")
                    return
                
                # Process data
                print(f'Processing data for {summoner_name}, {summoner_tag}, {server}')
                processed_data = self.process_data(summoner_name, summoner_tag, server, region_code)
                if processed_data is None:
                    return
                
                # Make predictions
                predictions = self.make_predictions(processed_data)
                if predictions is None:
                    return
                
                # Display results
                self.display_results(predictions)

if __name__ == "__main__":
    app = LoLPredictionApp(items_df = 'processed_data\items_df.csv', other_features_df = 'processed_data\other_features_df.csv', item_encoder = 'processed_data\item_encoder.pkl', champion_encoder = 'processed_data\champion_encoder.pkl')
    app.run()