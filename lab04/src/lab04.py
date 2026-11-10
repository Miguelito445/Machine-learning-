import os
import hashlib
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------------------------------------
# Step 1: Dataset Fetching & Problem Definition
# ---------------------------------------------------------
print("--- Step 1: Fetching Dataset & Defining Problem ---")
adult_data = fetch_openml(name='adult', version=2, as_frame=True)
df = adult_data.frame

# ---------------------------------------------------------
# Step 2: Data Ingestion Step
# ---------------------------------------------------------
print("--- Step 2: Data Ingestion ---")
raw_row_count, raw_col_count = df.shape
print(f"Ingested raw dataset - Row count: {raw_row_count}, Column count: {raw_col_count}")

target_col = 'class' if 'class' in df.columns else 'income'
if target_col != 'income' and 'income' not in df.columns:
    df.rename(columns={target_col: 'income'}, inplace=True)
    target_col = 'income'

# ---------------------------------------------------------
# Step 3: Validation Step
# ---------------------------------------------------------
print("--- Step 3: Data Validation ---")
for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

missing_question_marks = (df == '?').sum().sum()
null_counts = df.isnull().sum().sum()
print(f"Validation Report -> Missing/Unknown ('?') values detected: {missing_question_marks}")
print(f"Validation Report -> Standard Null values detected: {null_counts}")

age_violations = df[(df['age'] < 17) | (df['age'] > 90)].shape[0]
hours_violations = df[(df['hours-per-week'] < 1) | (df['hours-per-week'] > 99)].shape[0]
print(f"Validation Report -> Age range violations (17-90): {age_violations}")
print(f"Validation Report -> Hours-per-week range violations (1-99): {hours_violations}")

# ---------------------------------------------------------
# Step 4: Cleaning, Transformation & Stratified Split
# ---------------------------------------------------------
print("--- Step 4: Cleaning and Transformation Pipeline ---")
df.replace('?', np.nan, inplace=True)

X = df.drop(columns=[target_col])
y = df[target_col]
y_binary = y.astype(str).str.contains('>50K').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)

train_combined = X_train.copy()
train_combined['target'] = y_train.values
initial_train_len = len(train_combined)
train_combined.drop_duplicates(inplace=True)
print(f"Dropped {initial_train_len - len(train_combined)} duplicate rows from training set.")

X_train = train_combined.drop(columns=['target'])
y_train = train_combined['target']

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

quantiles_low = X_train[numeric_features].quantile(0.01)
quantiles_high = X_train[numeric_features].quantile(0.99)

def clip_outliers(df_in):
    df_out = df_in.copy()
    for col in numeric_features:
        df_out[col] = df_out[col].clip(lower=quantiles_low[col], upper=quantiles_high[col])
    return df_out

X_train_clipped = clip_outliers(X_train)
X_test_clipped = clip_outliers(X_test)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessor.fit(X_train_clipped)

os.makedirs('results', exist_ok=True)
joblib.dump(preprocessor, 'results/preprocessor.joblib')
print("Fitted preprocessor successfully saved to results/preprocessor.joblib")

# ---------------------------------------------------------
# Step 5: Feature Engineering
# ---------------------------------------------------------
print("--- Step 5: Feature Engineering ---")
def engineer_features(input_df):
    df_feat = input_df.copy()
    df_feat['capital_net'] = df_feat['capital-gain'] - df_feat['capital-loss']
    df_feat['hours_per_age'] = df_feat['hours-per-week'] / np.maximum(1, df_feat['age'] - 16)
    return df_feat

X_train_eng = engineer_features(X_train_clipped)
X_test_eng = engineer_features(X_test_clipped)

numeric_features_extended = numeric_features + ['capital_net', 'hours_per_age']

extended_preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features_extended),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline_base = Pipeline(steps=[('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
pipeline_base.fit(X_train_clipped, y_train)
preds_base = pipeline_base.predict(X_test_clipped)
print(f"Baseline (Without Engineering) -> Accuracy: {accuracy_score(y_test, preds_base):.4f}, F1: {f1_score(y_test, preds_base):.4f}")

pipeline_eng = Pipeline(steps=[('prep', extended_preprocessor), ('clf', LogisticRegression(max_iter=1000))])
pipeline_eng.fit(X_train_eng, y_train)
preds_eng = pipeline_eng.predict(X_test_eng)
print(f"Engineered (With Features)     -> Accuracy: {accuracy_score(y_test, preds_eng):.4f}, F1: {f1_score(y_test, preds_eng):.4f}")

# ---------------------------------------------------------
# Step 6: Label-Quality Check (Fixed with Sampling)
# ---------------------------------------------------------
print("--- Step 6: Label-Quality & Conflict Audit ---")
# Draw 50 random rows as specified in the lab instructions
sample_df = df.sample(n=50, random_state=42).copy()
sample_df['target_label'] = y_binary.loc[sample_df.index]
feature_cols = list(X.columns)

dup_mask = sample_df.duplicated(subset=feature_cols, keep=False)
if dup_mask.sum() > 0:
    duplicate_conflicts = sample_df[dup_mask].groupby(feature_cols, observed=True)['target_label'].nunique()
    conflict_count = (duplicate_conflicts > 1).sum()
else:
    conflict_count = 0

print(f"Label-Quality Audit (50 rows) -> Duplicate feature rows with conflicting labels found: {conflict_count}")

impossible_values = sample_df[(sample_df['age'] < 0) | (sample_df['hours-per-week'] < 0)].shape[0]
print(f"Label-Quality Audit -> Checked 50 random rows against constraints. Impossible values found: {impossible_values}")

# ---------------------------------------------------------
# Step 7: Artifact Generation (Hash, Config, Data Card)
# ---------------------------------------------------------
print("--- Step 7: Generating Hashes, Config, and Data Card ---")
X_test_final_processed = extended_preprocessor.fit_transform(X_test_eng)
processed_bytes = np.ascontiguousarray(X_test_final_processed).tobytes()
dataset_hash = hashlib.sha256(processed_bytes).hexdigest()
print(f"Processed Dataset SHA256 Hash: {dataset_hash}")

config_data = {
    'dataset': 'UCI Adult',
    'version': 2,
    'test_size': 0.30,
    'random_state': 42,
    'dataset_sha256': dataset_hash
}
with open('results/config.yaml', 'w') as f:
    yaml.dump(config_data, f)

data_card_content = f"""# Data Card: UCI Adult Dataset

- **Source**: UCI Machine Learning Repository via Scikit-Learn fetch_openml.
- **License**: Public Domain / Open Access.
- **Size**: {raw_row_count} total initial rows, {raw_col_count} columns.
- **Known Biases**: Historical demographic representation biases from US census data collection skewing towards specific regional distributions.
- **Privacy Concerns**: Contains sensitive demographic indicators (age, sex, native-country, marital status); anonymized records prevent direct individual identification.
- **Dataset Checksum (SHA256)**: `{dataset_hash}`
"""

with open('results/data_card.md', 'w') as f:
    f.write(data_card_content)

print("Pipeline execution complete! Config YAML, preprocessor, and data card successfully written to 'results/' directory.")
