import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# --- DATABASE CONFIGURATION ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/lendingclub")

def get_engine():
    """Returns the SQLAlchemy engine for database connections."""
    return create_engine(DB_URL)

# --- BUSINESS CONSTRAINTS ---
CAPITAL_LIMIT = 50000000   # ₹5 Crore
MAX_AVG_PD = 0.10          # 10% Risk Ceiling
MIN_MSME_FRAC = 0.015      # 1.5% MSME Quota

# --- INDIAN NUMBER FORMATTING ---
def format_inr(amount, decimals=0):
    """Formats a number in Indian numbering system (e.g., 5,00,00,000)."""
    is_negative = amount < 0
    amount = abs(amount)
    
    if decimals > 0:
        decimal_part = f".{round(amount % 1, decimals):.{decimals}f}"[1:]
    else:
        decimal_part = ""
    
    integer_part = str(int(amount))
    
    if len(integer_part) <= 3:
        result = integer_part
    else:
        result = integer_part[-3:]
        remaining = integer_part[:-3]
        while remaining:
            result = remaining[-2:] + ',' + result
            remaining = remaining[:-2]
    
    return ('-' if is_negative else '') + result + decimal_part