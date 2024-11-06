import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from LoLItemizationModel import LoLItemizationModel
import pickle
import json
from sklearn.preprocessing import LabelEncoder

class GetModelPredictions:
    def __init__(self, data):
        self.model_path = "lol_itemization_model.pth"  # Hardcoded path
        self.data = data

        with open('processed_data/item_encoder.pkl', 'rb') as f:
            self.item_encoder = pickle.load(f)

        with open('processed_data/champion_encoder.pkl', 'rb') as f:
            self.champion_encoder = pickle.load(f)

        self.items_df = pd.read_csv('processed_data/items_df.csv')       # Store these for use in get_item_name
        self.other_features_df = pd.read_csv('processed_data/other_features_df.csv')

        self.num_champions = len(self.champion_encoder.classes_)
        self.num_other_features = self.other_features_df.shape[1]
        self.output_dim = self.items_df.shape[1]
        with open('item.json') as f:
            self.item_names_df = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_inference_data(self, original_champion_name=None):
        
        # first we need to get original champion name that the player is playing and convert it to champion id from the champion encoder
        le = LabelEncoder()
        original_champion_name = [original_champion_name]
        self.data['champion_id'] = self.champion_encoder.transform(original_champion_name)[0]

        print(f"Champion ID: {self.data['champion_id']}")

        X_champion_inference = torch.LongTensor(self.data['champion_id'].values)
        X_other_inference = torch.FloatTensor(self.data.drop('champion_id', axis=1).values)
        

        print(f"Other features tensor shape in prepare_inference_data: {X_other_inference.shape}")
        
        inference_data = TensorDataset(X_champion_inference, X_other_inference)
        
        inference_loader = DataLoader(inference_data, batch_size=1, shuffle=False)

        print(f'Inference Loader: {inference_loader}')
        
        return inference_loader

    def load_model(self):

        model = LoLItemizationModel(
                self.num_champions, 
                self.num_other_features, 
                output_dim=self.output_dim
            )
        
        # Load the state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        model.load_state_dict(state_dict)
            
        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()
            
        return model

    def get_predictions(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            try:
                # Fetching data from the test_loader
                champion_ids, other_features = next(iter(test_loader))
                print("Other Features Shape in get predictions:", other_features.shape)  # Print shape of other features

                # Moving data to the appropriate device
                champion_ids = champion_ids.to(self.device)
                other_features = other_features.to(self.device)

                print("moving to outputs")  
                # Forward pass through the model
                outputs = model(champion_ids, other_features)
                print("Model Outputs:", outputs)  # Print model outputs to inspect values

                # Convert outputs to numpy
                predicted_items = outputs.cpu().numpy()
                print("Predicted Items (numpy):", predicted_items)  # Print numpy array of predicted items

                # Get the indices of the top 7 items
                top_7_items = np.argsort(predicted_items[0])[::-1][:7]
                print("Top 7 Item Indices:", top_7_items)  # Print indices of the top 7 items

                # Map indices to item names
                top_7_items_names = [
                    self.get_item_name(item) 
                    for item in top_7_items
                ]
                print("Top 7 Item Names:", top_7_items_names)  # Print the names of the top 7 items
                
                return top_7_items_names

            except Exception as e:
                print("Error in get_predictions:", str(e))
                return None
            

    def get_item_name(self, item_column_index):
        """
        Get the item name from the column index in the encoded items DataFrame
        
        Args:
            item_column_index (int): The index of the item column in the encoded DataFrame
            
        Returns:
            str: The name of the item from the item_names_df
        """
        try:
            # Get the actual item ID from the encoder classes
            # item_encoder.classes_ contains the original item IDs in order
            item_id = self.item_encoder.classes_[item_column_index]
            
            # Get the item name from the item_names_df
            item_name = self.item_names_df['data'][str(item_id)]['name']
            
            return item_name
            
        except IndexError:
            print(f"Error: Column index {item_column_index} is out of bounds. Max index is {len(self.item_encoder.classes_) - 1}")
            return None
        except KeyError:
            print(f"Error: Item ID {item_id} not found in item_names_df")
            return None

    def predict(self, data, original_champion_name):  # Changed signature to accept single parameter
        """Main prediction pipeline"""
        self.data = data  # Update the data
        
        print(f"Data in predict: {self.data}, shape: {self.data.shape}")
        # Prepare data
        inference_loader = self.prepare_inference_data(original_champion_name)
        if inference_loader is None:
            return None
        # Load model

        print("Loading model...")
        model = self.load_model()
        if model is None:
            return None

        print("Model loaded successfully")
        # Get predictions
        predictions = self.get_predictions(model, inference_loader)
        return predictions