import pandas as pd
import time
from tqdm import tqdm
from config import get_engine

# 1. The Data File
csv_file = 'loan.csv' 

# 2. The PostgreSQL Connection
engine = get_engine()

# 3. The Ingestion Engine
chunk_size = 10000
start_time = time.time()

print("Connecting to PostgreSQL and starting data ingestion...")

# Read everything as strings to prevent data type clashes
chunk_iterator = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False, dtype=str)

# We use 'enumerate' to count the chunks as they process (0, 1, 2, 3...)
for i, chunk in enumerate(tqdm(chunk_iterator, desc="Ingesting 10k-row chunks")):
    
    if i == 0:
        # FOR CHUNK 0: 'replace' completely destroys the old table and creates a fresh, all-text schema
        chunk.to_sql(name='raw_loan_data', con=engine, if_exists='replace', index=False, chunksize=1000)
    else:
        # FOR ALL OTHER CHUNKS: 'append' adds the data to our new, clean table
        chunk.to_sql(name='raw_loan_data', con=engine, if_exists='append', index=False, chunksize=1000)

print(f"\nData ingestion complete in {round((time.time() - start_time) / 60, 2)} minutes.")