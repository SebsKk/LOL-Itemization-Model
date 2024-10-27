from torch import nn
import torch


class LoLItemizationModel(nn.Module):
    def __init__(self, num_champions, num_other_features, embedding_dim=32, hidden_dim=64, output_dim=100):
        super(LoLItemizationModel, self).__init__()
        
        # Champion embedding layer
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        
        # Layers for processing champion embeddings
        self.champion_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Layers for processing other features and combining with champion features
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_dim + num_other_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, champion_ids, other_features):
        # Process champion embeddings
        champion_embedded = self.champion_embedding(champion_ids)
        champion_features = self.champion_layers(champion_embedded)
        
        # Combine champion features with other features
        combined = torch.cat([champion_features, other_features], dim=1)
        
        # Process combined features
        output = self.combined_layers(combined)
        
        return output