import pandas as pd
from pulp import *
from config import get_engine, CAPITAL_LIMIT, MIN_MSME_FRAC, format_inr

# Database connection
engine = get_engine()

# === STEP 1: LOAD DATA FIRST ===
print("Loading data...")
df = pd.read_sql("""
    SELECT f.loan_id, f.loan_amnt, f.purpose, f.expected_interest, p.prob_default
    FROM loan_features f
    JOIN loan_predictions p ON f.loan_id = p.loan_id
    WHERE f.loan_amnt IS NOT NULL 
      AND p.prob_default IS NOT NULL
    LIMIT 5000
""", engine)

print(f"Loaded {len(df):,} loan candidates")

# === STEP 2: NOW YOU CAN USE df ===
# Define MSME purposes based on YOUR data
MSME_PURPOSES = ['small_business']
df['is_msme'] = df['purpose'].isin(MSME_PURPOSES)

# Feasibility check (NOW df exists!)
available_msme_capital = df.loc[df['is_msme'], 'loan_amnt'].sum()
total_available_capital = df['loan_amnt'].sum()
min_msme_required = MIN_MSME_FRAC * CAPITAL_LIMIT

print(f"\n📊 MSME Feasibility Check:")
print(f"Available MSME capital: ₹{format_inr(available_msme_capital)}")
print(f"Required MSME capital ({MIN_MSME_FRAC:.1%}): ₹{format_inr(min_msme_required)}")

if available_msme_capital < min_msme_required:
    print(f"\n⚠️ Adjusting MSME target to feasible level...")
    MIN_MSME_FRAC = min(MIN_MSME_FRAC, available_msme_capital / CAPITAL_LIMIT)
    print(f"New MSME target: {MIN_MSME_FRAC:.1%}")