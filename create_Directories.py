import os

# Define the base directory
base_dir = r"C:\EngineRoom"

# List of directories to create relative to the base_dir
folders = [
    "data/raw",
    "data/processed",
    "data/synthetic",
    "data/backups",
    "models",
    "code/setup",
    "code/data_ingestion",
    "code/preprocessing",
    "code/training",
    "code/evaluation",
    "code/visualization",
    "logs",
    "docs",
    "experiments",
    "config"
]

# Create each folder if it doesn't exist
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created or verified directory: {path}")
