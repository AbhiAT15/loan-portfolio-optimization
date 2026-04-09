import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config import get_engine, CAPITAL_LIMIT, format_inr
import warnings
warnings.filterwarnings('ignore')

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="CRO Portfolio Dashboard", layout="wide")
st.title("Executive Loan Portfolio Optimization")
st.markdown("---")

# --- 2. DATA CONNECTION ---
@st.cache_data
def load_data():
    engine = get_engine()
    query = "SELECT * FROM optimization_results;"
    return pd.read_sql(query, engine)

try:
    df = load_data()
except Exception as e:
    st.error(f"Database connection failed. Did you update the password? Error: {e}")
    st.stop()

# --- 3. CALCULATE KPIs ---
approved_df = df[df['approved'] == 1].copy()
rejected_df = df[df['approved'] == 0].copy()

# Metrics
total_budget = CAPITAL_LIMIT
capital_deployed = approved_df['loan_amnt'].sum()

# Re-calculate the True Capital-Weighted Adjusted Risk (CGTMSE logic)
if capital_deployed > 0:

    approved_df['adjusted_pd'] = np.where(
        approved_df['purpose'] == 'small_business',
        approved_df['prob_default'] * 0.20,  # Condition is True: Apply 80% guarantee
        approved_df['prob_default']          # Condition is False: Keep raw risk
    )

    total_risk_weight = (approved_df['adjusted_pd'] * approved_df['loan_amnt']).sum()
    avg_pd = total_risk_weight / capital_deployed
    
    #Calculate Net Risk-Adjusted Profit
    approved_df['risk_adjusted_profit'] = approved_df['expected_interest'] * (1 - approved_df['adjusted_pd'])
    expected_profit = approved_df['risk_adjusted_profit'].sum()
else:
    avg_pd = 0
    expected_profit = 0

# --- 4. EXECUTIVE SUMMARY METRICS (Top Row) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Capital Deployed", f"₹{format_inr(capital_deployed)} / ₹{format_inr(CAPITAL_LIMIT)}")
col2.metric("Expected Profit", f"₹{format_inr(expected_profit)}")
col3.metric("Adjusted Portfolio Risk", f"{avg_pd * 100:.2f}%", "Ceiling: 10.00%", delta_color="inverse")
col4.metric("Approval Rate", f"{(len(approved_df) / len(df)) * 100:.1f}%")

st.markdown("---")

# --- 5. VISUALIZATIONS ---
colA, colB = st.columns(2)

with colA:
    st.subheader("Capital Allocation Status")
    allocation_data = pd.DataFrame({
        "Status": ["Deployed", "Unused Buffer"],
        "Amount": [capital_deployed, total_budget - capital_deployed]
    })
    fig_pie = px.pie(allocation_data, values="Amount", names="Status", hole=0.4, color_discrete_sequence=['#2ecc71', '#bdc3c7'])
    st.plotly_chart(fig_pie, use_container_width=True)

with colB:
    st.subheader("Risk vs. Profit Distribution")
    fig_scatter = px.scatter(
        df, 
        x="prob_default", 
        y="expected_interest", 
        color="approved",
        labels={
            "prob_default": "Raw Probability of Default", 
            "expected_interest": "Expected Profit (₹)",
            "approved": "Decision (1=Approve)"
        },
        color_continuous_scale=['#e74c3c', '#2ecc71'], 
        opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 6. RAW LEDGER ---
st.markdown("---")
st.subheader("Approved Loan Ledger")

display_cols = ['loan_id', 'purpose', 'loan_amnt', 'expected_interest', 'prob_default']
st.dataframe(
    approved_df[display_cols]
    .style.format({
        'loan_amnt': lambda x: f'₹{format_inr(x, 2)}',
        'expected_interest': lambda x: f'₹{format_inr(x, 2)}',
        'prob_default': '{:.2%}'
    }),
    use_container_width=True
)