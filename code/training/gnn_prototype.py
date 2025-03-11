# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os  # Import os to create directories

# ===============================
# Step 1: Load Synthetic Data
# ===============================
# Define the path to the synthetic data file generated in Phase 0
data_file = r"C:\EngineRoom\data\synthetic\synthetic_posts.parquet"

# Load the synthetic data using Pandas
df = pd.read_parquet(data_file)

# Extract numerical features: likes, shares, comments
features = df[['likes', 'shares', 'comments']].values.astype(np.float32)

# Normalize features to have zero mean and unit variance
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Get the total number of nodes (posts)
num_nodes = features.shape[0]

# ===============================
# Step 2: Create a Dummy Graph
# ===============================
# For demonstration, create a sparse random graph:
# We'll add an edge between nodes with a very low probability
edge_index = []
np.random.seed(42)  # For reproducibility
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if np.random.rand() < 0.001:  # Adjust sparsity as needed
            edge_index.append([i, j])
            edge_index.append([j, i])
# Convert edge list to a PyTorch tensor (transpose for correct shape)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create a node feature tensor from the normalized features
x = torch.tensor(features, dtype=torch.float)

# ===============================
# Step 3: Generate Dummy Labels
# ===============================
# For demonstration, generate random sentiment labels (3 classes: 0, 1, 2)
num_classes = 3
y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)

# ===============================
# Step 4: Create PyG Data Object
# ===============================
# This object holds our graph structure, node features, and labels
data = Data(x=x, edge_index=edge_index, y=y)

# ===============================
# Step 5: Define the GNN Model
# ===============================
# A simple two-layer Graph Convolutional Network (GCN) for node classification
class GNNPrototype(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNPrototype, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)  # First GCN layer
        self.conv2 = GCNConv(hidden_channels, num_classes)     # Output GCN layer
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # Apply first convolution
        x = F.relu(x)                # Activation function
        x = self.conv2(x, edge_index)  # Apply second convolution to get logits
        return x

# Instantiate the model with appropriate dimensions
model = GNNPrototype(num_features=x.size(1), hidden_channels=16, num_classes=num_classes)

# Set up the optimizer and loss function for training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ===============================
# Step 6: Train the GNN Model
# ===============================
def train():
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Reset gradients
    out = model(data)      # Forward pass: compute logits
    loss = criterion(out, data.y)  # Compute loss against true labels
    loss.backward()        # Backpropagate the error
    optimizer.step()       # Update model weights
    return loss.item()

# Train for 20 epochs
for epoch in range(1, 21):
    loss = train()
    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# ===============================
# Step 7: Save the Model Checkpoint
# ===============================
# Define the path to save the checkpoint
checkpoint_path = r"C:\EngineRoom\models\gnn\gnn_prototype.pth"

# Ensure the parent directory exists (this is the modification)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Now save the model checkpoint
torch.save(model.state_dict(), checkpoint_path)
print(f"Model checkpoint saved to {checkpoint_path}")
