# Step 1: Connect to PostgreSQL
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # NEW: The Imputer to fix blank cells
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import time
from config import get_engine

# Database connection
engine = get_engine()

print("1. Loading data from PostgreSQL...")
df = pd.read_sql("SELECT * FROM loan_features;", engine)
print(f"Loaded {df.shape[0]:,} rows.")

# Step 2: Prepare features (X) and target (y)
X = df.drop(['loan_id', 'is_default'], axis=1)
y = df['is_default']

# Convert the 'purpose' text into numbers
X = pd.get_dummies(X, columns=['purpose'], drop_first=True)

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n2. Imputing Missing Data & Scaling...")
# THE FIX: Mathematically estimate missing values using the median instead of dropping them
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale features so Income and Interest Rate are weighted fairly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("\n3. Training Logistic Regression Model...")
start_time = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
print(f"Training Complete in {round(time.time() - start_time, 2)} seconds.")

print("\n4. Evaluating Model...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\n*** Model AUC-ROC Score: {auc_score:.4f} ***\n")

print("5. Generating Out-of-Fold (unbiased) probabilities for the entire portfolio...")
# Re-impute/scale over the full dataset properly 
X_full_imputed = imputer.fit_transform(X)
X_full_scaled = scaler.fit_transform(X_full_imputed)

# Create Out-of-Fold predictions so that we aren't optimizing on in-sample (biased) data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['prob_default'] = cross_val_predict(model, X_full_scaled, y, cv=cv, method='predict_proba')[:, 1]

# Create results table and push to PostgreSQL
results_df = df[['loan_id', 'prob_default']]
results_df.to_sql('loan_predictions', engine, if_exists='replace', index=False)

print(f"✅ Saved predictions for {len(df):,} loans to 'loan_predictions' table in PostgreSQL.")