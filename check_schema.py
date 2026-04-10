import pandas as pd
from config import get_engine

engine = get_engine()
try:
    df = pd.read_sql("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'raw_loan_data';", engine)
    print(df['column_name'].values)
except Exception as e:
    print(e)
