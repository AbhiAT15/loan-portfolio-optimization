import pandas as pd
from pulp import *
import time

# --- NEW SECURE IMPORTS ---
from config import get_engine, CAPITAL_LIMIT, MAX_AVG_PD, MIN_MSME_FRAC, format_inr

# 1. Database Connection (Now perfectly secure)
engine = get_engine()

print("1. Loading applicant batch from PostgreSQL...")
df = pd.read_sql("""
    SELECT f.loan_id, f.loan_amnt, f.purpose, f.expected_interest, p.prob_default
    FROM loan_features f
    JOIN loan_predictions p ON f.loan_id = p.loan_id
    WHERE f.loan_amnt IS NOT NULL 
      AND p.prob_default IS NOT NULL
    ORDER BY loan_id
    LIMIT 15000
""", engine)

print(f"Optimizing over {len(df):,} loan candidates...")

# 3. Build the Linear Programming Problem
print("2. Initializing the PuLP Linear Solver...")
start_time = time.time()
prob = LpProblem("Loan_Portfolio_Optimization", LpMaximize)

n = len(df)
x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

loan_amnt  = df['loan_amnt'].tolist()
raw_pd     = df['prob_default'].tolist()
purpose    = df['purpose'].tolist()

# --- THE FINANCIAL ENGINEERING (CGTMSE GUARANTEE) ---
# Apply 80% government backstop to MSMEs. Bank only holds 20% of the risk.
adjusted_pd = [raw_pd[i] * 0.20 if purpose[i] == 'small_business' else raw_pd[i] for i in range(n)]

# Objective: Maximize expected profit, using the adjusted survival rate
exp_profit = [df.loc[i, 'expected_interest'] * (1 - adjusted_pd[i]) for i in range(n)]
prob += lpSum(exp_profit[i] * x[i] for i in range(n))

# Constraint 1: Do not exceed ₹5 Crore Capital Limit
prob += lpSum(loan_amnt[i] * x[i] for i in range(n)) <= CAPITAL_LIMIT

# Constraint 2: Maintain Portfolio Risk <= 10% (Using the Adjusted PD)
prob += lpSum((adjusted_pd[i] - MAX_AVG_PD) * loan_amnt[i] * x[i] for i in range(n)) <= 0

# Constraint 3: Ensure 1.5% of capital goes to 'small_business'
prob += lpSum(loan_amnt[i] * x[i] for i in range(n) if purpose[i] == 'small_business') >= \
       MIN_MSME_FRAC * lpSum(loan_amnt[i] * x[i] for i in range(n))

# 4. Solve the Math
print("3. Solving Constraints...")
prob.solve(PULP_CBC_CMD(msg=0))
print(f"Solver finished in {round(time.time() - start_time, 2)} seconds.")

# 5. Output Results
print(f"\nOptimization Status: {LpStatus[prob.status]}")

if LpStatus[prob.status] == 'Optimal':
    approved = [i for i in range(n) if value(x[i]) == 1]
    approved_df = df.iloc[approved].copy()

    total_capital = approved_df['loan_amnt'].sum()
    
    # Calculate the True Capital-Weighted Adjusted Risk
    total_risk_weight = sum(adjusted_pd[i] * loan_amnt[i] for i in approved)
    weighted_avg_pd = total_risk_weight / total_capital if total_capital > 0 else 0
    msme_capital = approved_df[approved_df['purpose'] == 'small_business']['loan_amnt'].sum()

    print("-" * 45)
    print("FINAL OPTIMIZED PORTFOLIO (WITH CGTMSE)")
    print("-" * 45)
    print(f"Applicants Evaluated:  {n:,}")
    print(f"Loans Approved:        {len(approved_df):,}")
    print(f"Total Capital Deployed: ₹{format_inr(total_capital, 2)} / ₹{format_inr(CAPITAL_LIMIT, 2)}")
    print(f"Total Expected Profit:  ₹{format_inr(sum(exp_profit[i] for i in approved), 2)}")
    print(f"Adjusted Portfolio Risk:{weighted_avg_pd:.2%}")
    print(f"MSME Capital Share:     {msme_capital / total_capital:.2%}")
    print("-" * 45)

    # Save results to database (Overwriting the old failed run)
    approved_df['approved'] = 1
    rejected_df = df.iloc[[i for i in range(n) if i not in approved]].copy()
    rejected_df['approved'] = 0
    final = pd.concat([approved_df, rejected_df])
    final.to_sql('optimization_results', engine, if_exists='replace', index=False)
    print("\n✅ Success: Engineered portfolio saved to 'optimization_results' in PostgreSQL.")
else:
    print("FAILED: The solver could not find a valid mathematical portfolio.")