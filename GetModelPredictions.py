import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from LoLItemizationModel import LoLItemizationModel
import pickle
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_inference_data(self):

        X_champion_inference = torch.LongTensor(self.data['champion_id'].values)
        X_other_inference = torch.FloatTensor(self.data.drop('champion_id', axis=1).values)
        
        print(f"Champion tensor shape: {X_champion_inference.shape}")
        print(f"Other features tensor shape: {X_other_inference.shape}")
        
        inference_data = TensorDataset(X_champion_inference, X_other_inference)
        
        inference_loader = DataLoader(inference_data, batch_size=1, shuffle=False)
        
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
                champion_ids, other_features = next(iter(test_loader))
                
                champion_ids = champion_ids.to(self.device)
                other_features = other_features.to(self.device)

                outputs = model(champion_ids, other_features)
                predicted_items = outputs.cpu().numpy()
            
                # Get top 7 items
                top_7_items = np.argsort(predicted_items[0])[::-1][:7]
                top_7_items_names = [
                    self.get_item_name(item) 
                    for item in top_7_items
                ]
                return top_7_items_names
            

    def get_item_name(self, item_number):  # Remove extra parameters
        # Create a binary vector for the single item
        binary_vector = np.zeros((1, len(self.item_encoder.classes_)))
        binary_vector[0, item_number] = 1
        
        # Use inverse_transform on the binary vector
        item_id = self.item_encoder.inverse_transform(binary_vector)[0][0]
        item_name = self.items_df['data'][str(item_id)]['name']
        return item_name

    def predict(self, data):  # Changed signature to accept single parameter
        """Main prediction pipeline"""
        self.data = data  # Update the data
        
        # Prepare data
        inference_loader = self.prepare_inference_data()
        if inference_loader is None:
            return None

        # Load model
        model = self.load_model()
        if model is None:
            return None

        # Get predictions
        predictions = self.get_predictions(model, inference_loader)
        return predictions