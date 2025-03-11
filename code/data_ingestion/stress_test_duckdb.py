import duckdb
import pandas as pd
import os

# Define paths for the synthetic data and the DuckDB database
synthetic_data_path = r"C:\EngineRoom\data\synthetic\synthetic_posts.parquet"
duckdb_path = r"C:\EngineRoom\data\raw\engine_data.duckdb"

# Connect to (or create) the DuckDB database
con = duckdb.connect(database=duckdb_path, read_only=False)
print(f"Connected to DuckDB database at {duckdb_path}")

# Read synthetic data in chunks to simulate large-scale ingestion
chunk_size = 1000  # Number of rows per chunk

# Use Pandas to read the Parquet file in chunks
# Note: Pandas doesn't support chunking for Parquet directly, so we simulate it:
df = pd.read_parquet(synthetic_data_path)
num_chunks = len(df) // chunk_size + 1

for i in range(num_chunks):
    # Define chunk boundaries
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    chunk = df.iloc[start_idx:end_idx]
    
    # Append chunk to DuckDB table
    chunk.to_sql("posts", con, if_exists="append", index=False)
    print(f"Ingested chunk {i+1}/{num_chunks} with rows {start_idx} to {end_idx}")

# Perform a sample query to test retrieval speed
query = "SELECT COUNT(*) FROM posts"
result = con.execute(query).fetchall()
print(f"Total posts ingested: {result[0][0]}")

# Close the connection
con.close()
print("DuckDB ingestion test completed successfully.")
