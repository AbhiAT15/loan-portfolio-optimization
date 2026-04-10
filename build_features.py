import pandas as pd
from sqlalchemy import text
from config import get_engine
import time

def build_features():
    engine = get_engine()
    print("1. Loading raw expected features from PostgreSQL 'raw_loan_data'...")
    start_time = time.time()
    
    # We will pick a reasonable chunk of loans to process for this prototype
    # Instead of all 887k, let's load a subset that has sufficient populated values
    # Or we can load them all. We use chunksize if memory is a concern, but typically 1GB is fine.
    
    query = """
    SELECT 
        id as loan_id,
        loan_amnt,
        term,
        int_rate,
        installment,
        grade,
        emp_length,
        home_ownership,
        annual_inc,
        verification_status,
        loan_status,
        purpose,
        dti,
        delinq_2yrs,
        pub_rec,
        revol_util
    FROM raw_loan_data
    """
    
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} raw rows in {round(time.time() - start_time, 2)} seconds.")
    
    print("2. Formatting types and constructing features...")
    # Convert numerical strings to floats where necessary
    numeric_cols = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'pub_rec']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean percentages
    df['int_rate'] = df['int_rate'].astype(str).str.replace(r'%', '', regex=True)
    df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
    
    df['revol_util'] = df['revol_util'].astype(str).str.replace(r'%', '', regex=True)
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
    
    # Parse term months
    df['term_months'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Engineered Feature: Expected Gross Interest (User designated heuristic)
    df['expected_interest'] = (df['installment'] * df['term_months']) - df['loan_amnt']
    
    # Classification logic for 'is_default' (Binary)
    # E.g. Default, Charged Off, Late (31-120 days)
    bad_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
    df['is_default'] = df['loan_status'].isin(bad_statuses).astype(int)
    
    # Map categorical grades to numeric levels (A=1, B=2...)
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_num'] = df['grade'].map(grade_mapping)
    
    # Map home_ownership to numeric roughly reflecting stability
    home_mapping = {'MORTGAGE': 3, 'OWN': 2, 'RENT': 1}
    df['home_ownership_num'] = df['home_ownership'].map(home_mapping).fillna(0)
    
    # Map verification_status
    vf_mapping = {'Verified': 2, 'Source Verified': 1, 'Not Verified': 0}
    df['verification_num'] = df['verification_status'].map(vf_mapping).fillna(0)
    
    print("3. Finalizing output table and committing to database...")
    # Select final columns to match the established pipeline
    final_cols = [
        'loan_id', 'int_rate', 'installment', 'annual_inc', 'dti', 
        'delinq_2yrs', 'pub_rec', 'revol_util', 'term_months', 'loan_amnt', 
        'is_default', 'expected_interest', 'grade_num', 
        'home_ownership_num', 'verification_num', 'purpose'
    ]
    
    df_features = df[final_cols].copy()
    
    # Drop rows where loan_id is completely null
    df_features = df_features.dropna(subset=['loan_id'])
    
    # Remove duplicates if any
    df_features = df_features.drop_duplicates(subset=['loan_id'])
    
    start_time = time.time()
    # Save feature table out
    df_features.to_sql('loan_features', engine, if_exists='replace', index=False)
    
    # Recommend creating an index for optimization step speed
    with engine.connect() as con:
        # Check if index exists or just gracefully create it.
        try:
            con.execute(text("CREATE INDEX idx_loan_features_loan_id ON loan_features (loan_id);"))
        except:
            pass # Index might already exist
            
    print(f"✅ Success! Wrote {len(df_features):,} rows to 'loan_features' in {round(time.time() - start_time, 2)} seconds.")

if __name__ == "__main__":
    build_features()
