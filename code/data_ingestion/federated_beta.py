# federated_beta.py
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define a simple GNN model that matches the one used during training
class GNNPrototype(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNPrototype, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)  # First GCN layer
        self.conv2 = GCNConv(hidden_channels, num_classes)     # Output GCN layer
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First convolution
        x = F.relu(x)                # ReLU activation
        x = self.conv2(x, edge_index)  # Second convolution
        return x

# ===============================
# Step 2: Load User Data
# ===============================
# Expecting a CSV file with columns: post_id, likes, shares, comments
# For demonstration, we will use a default file path if not provided via command-line
input_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\EngineRoom\code\data_ingestion\user_posts.csv"

try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading input file {input_file}: {e}")
    sys.exit(1)

# Check if required columns are present
required_columns = ['post_id', 'likes', 'shares', 'comments']
if not all(col in df.columns for col in required_columns):
    print(f"Input file must contain the following columns: {required_columns}")
    sys.exit(1)

# ===============================
# Step 3: Preprocess Data
# ===============================
# Extract and normalize numerical features (likes, shares, comments)
features = df[['likes', 'shares', 'comments']].values.astype(np.float32)
features = (features - features.mean(axis=0)) / features.std(axis=0)

num_nodes = features.shape[0]
x = torch.tensor(features, dtype=torch.float)

# For inference, we create a trivial graph with no connections (isolated nodes)
# This is done by providing an empty edge_index with the correct shape.
edge_index = torch.empty((2, 0), dtype=torch.long)

# Create PyG Data object for inference
data = Data(x=x, edge_index=edge_index)

# ===============================
# Step 4: Load the Trained Model
# ===============================
# Define model parameters (should match those used in training)
num_features = x.size(1)
hidden_channels = 16
num_classes = 3
model = GNNPrototype(num_features, hidden_channels, num_classes)

# Load model checkpoint (ensure the path is correct)
checkpoint_path = r"C:\EngineRoom\models\gnn\gnn_prototype.pth"
if not os.path.exists(checkpoint_path):
    print(f"Model checkpoint not found at {checkpoint_path}. Exiting.")
    sys.exit(1)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # Set model to evaluation mode

# ===============================
# Step 5: Run Inference and Generate Recommendations
# ===============================
# Run the model to get predictions (logits)
with torch.no_grad():
    logits = model(data)
    predictions = torch.argmax(logits, dim=1).numpy()

# Map predictions to simple recommendations
# For demonstration: 0 => "Revise tone", 1 => "Maintain content", 2 => "High engagement: post now"
recommendation_map = {
    0: "Revise tone for better engagement",
    1: "Content is okay, consider minor adjustments",
    2: "High engagement predicted: Post immediately"
}

# Build a list of recommendations for each post
results = []
for idx, row in df.iterrows():
    post_id = row['post_id']
    pred = int(predictions[idx])
    recommendation = recommendation_map.get(pred, "No recommendation")
    results.append({
        "post_id": post_id,
        "predicted_sentiment_class": pred,
        "recommendation": recommendation
    })

# ===============================
# Step 6: Save Recommendations to JSON
# ===============================
# Define the output file path
output_file = r"C:\EngineRoom\experiments\recommendations.json"
# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write the recommendations to the JSON file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Recommendations saved to {output_file}")
