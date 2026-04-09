# Retail Loan Portfolio Optimization under Capital Constraints

A **full-stack data science pipeline** that optimizes a retail loan portfolio using Linear Programming, balancing profitability against regulatory risk constraints — built on real Lending Club data (887K+ loan records).

---

## Business Problem

A retail bank has a **₹5 Crore (₹5,00,00,000) capital budget** to deploy as loans. The challenge: maximize expected profit while satisfying three competing constraints:

| Constraint | Threshold | Rationale |
|---|---|---|
| **Capital Budget** | ≤ ₹5,00,00,000 | Hard limit on deployable capital |
| **Portfolio Default Risk** | ≤ 10% (weighted avg PD) | CRO risk ceiling |
| **MSME Allocation** | ≥ 1.5% of capital | Priority sector lending mandate |

### CGTMSE Financial Engineering
MSME loans backed by the [Credit Guarantee Fund Trust](https://www.cgtmse.in/) receive an **80% government guarantee**, meaning the bank retains only 20% of the default risk. This is modeled as an adjusted probability of default:

```
Adjusted PD = Raw PD × 0.20  (for small_business loans)
Adjusted PD = Raw PD          (for all other loans)
```

This guarantee makes MSME loans significantly more attractive on a risk-adjusted basis, enabling the optimizer to include them without breaching the risk ceiling.

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐     ┌────────────────┐
│  loan.csv   │────▸│  PostgreSQL  │────▸│  Logistic Regression│────▸│  LP Optimizer  │
│  (887K rows)│     │  (Raw + FE)  │     │  (PD Prediction)    │     │  (PuLP/CBC)    │
└─────────────┘     └──────────────┘     └─────────────────────┘     └───────┬────────┘
                                                                             │
                                                                    ┌────────▼────────┐
                                                                    │   Streamlit     │
                                                                    │   Dashboard     │
                                                                    └─────────────────┘
```

| Phase | Script | Description |
|---|---|---|
| **1. Ingest** | `ingest_loan_data.py` | Chunked CSV → PostgreSQL ingestion (handles 1.1GB file) |
| **2. Train** | `train_risk_model.py` | Logistic Regression with median imputation + standard scaling |
| **3. Optimize** | `portfolio_optimizer.py` | Binary LP: maximize risk-adjusted profit subject to constraints |
| **4. Visualize** | `dashboard.py` | Executive Streamlit dashboard with KPIs + Plotly charts |

---

## Key Technical Highlights

- **Linear Programming** (PuLP + CBC solver) for combinatorial portfolio selection over 15,000 loan candidates
- **Linearized risk constraint**: `Σ (PD_i - MAX_PD) × Amount_i × x_i ≤ 0` avoids nonlinear ratio formulation
- **CGTMSE guarantee modeling** as an adjusted PD multiplier within the LP objective
- **Centralized configuration** (`config.py`) with environment-variable-based secrets management
- **Vectorized Pandas/NumPy** operations (e.g., `np.where` for CGTMSE adjustment in dashboard)
- **Indian number formatting** (`format_inr`) for currency display (₹5,00,00,000 style)

---

## Setup & Usage

### Prerequisites
- Python 3.8+
- PostgreSQL (running locally)
- Lending Club loan dataset (`loan.csv` from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club))

### Installation

```bash
# Clone the repository
git clone https://github.com/AbhiAT15/loan-portfolio-optimization.git
cd loan-portfolio-optimization

# Install dependencies
pip install -r requirements.txt

# Configure your database connection
# Create a .env file with your PostgreSQL credentials:
echo DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/lendingclub > .env
```

### Run the Pipeline

```bash
# Step 1: Ingest raw data into PostgreSQL
python ingest_loan_data.py

# Step 2: Train the risk model and generate PD predictions
python train_risk_model.py

# Step 3: Run the portfolio optimizer
python portfolio_optimizer.py

# Step 4: Launch the executive dashboard
streamlit run dashboard.py
```

---

## Project Structure

```
├── config.py                  # Centralized DB connection & business constants
├── ingest_loan_data.py        # CSV → PostgreSQL chunked ingestion
├── train_risk_model.py        # Logistic Regression PD model
├── portfolio_optimizer.py     # PuLP Linear Programming solver
├── dashboard.py               # Streamlit executive dashboard
├── diagnostic.py              # Utility: inspect loan purpose distribution
├── check_default_distribution.sql # Utility: check target class distribution
├── temp_feasibility_check.py  # Utility: MSME constraint feasibility analysis
├── requirements.txt           # Python dependencies
├── .env                       # Local DB credentials (not tracked in git)
├── .gitignore                 # Excludes data files, secrets, caches
└── README.md                  # This file
```

---

## Dashboard Preview

![Executive Dashboard](Streamlit_dashboard_v2.png)

The Streamlit dashboard provides:
- **Executive KPI cards**: Capital deployed, expected profit, adjusted portfolio risk, approval rate
- **Capital allocation donut chart**: Deployed vs. unused buffer
- **Risk vs. profit scatter plot**: Visualizes the optimizer's approve/reject decisions
- **Approved loan ledger**: Detailed table with formatted currency and percentage columns

---

## Data Source

[Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — 887,379 loan records with 75 features including loan amount, interest rate, purpose, annual income, and repayment status.

---

## Tech Stack

`Python` · `PostgreSQL` · `SQLAlchemy` · `scikit-learn` · `PuLP (CBC Solver)` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

## License

This project is for educational and portfolio demonstration purposes.
