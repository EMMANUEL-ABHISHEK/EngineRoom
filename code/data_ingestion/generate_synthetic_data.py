import pandas as pd
from faker import Faker
import random
import os

# Initialize Faker for generating synthetic data
fake = Faker()

# Define number of synthetic posts to generate
num_posts = 10000

# Function to generate a single synthetic post
def generate_post():
    return {
        "post_id": fake.uuid4(),                  # Unique post identifier
        "timestamp": fake.date_time_this_year(),    # Random datetime within this year
        "user": fake.user_name(),                   # Simulated username
        "content": fake.sentence(nb_words=10),       # Random post content
        "likes": random.randint(0, 1000),            # Random like count
        "shares": random.randint(0, 500),            # Random share count
        "comments": random.randint(0, 300)           # Random comment count
    }

# Generate the synthetic dataset
data = [generate_post() for _ in range(num_posts)]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data)

# Define the output directory for synthetic data
output_dir = r"C:\EngineRoom\data\synthetic"
output_file = os.path.join(output_dir, "synthetic_posts.parquet")

# Save DataFrame as a Parquet file with Zstandard compression for efficiency
df.to_parquet(output_file, compression="zstd")
print(f"Synthetic data generated and saved to {output_file}")
