import pandas as pd
from config import get_engine

engine = get_engine()

print("Fetching unique loan purposes...")
query = "SELECT purpose, COUNT(*) as count FROM loan_features GROUP BY purpose ORDER BY count DESC;"
df = pd.read_sql(query, engine)

print(df)