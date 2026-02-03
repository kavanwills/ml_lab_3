# %%

# Step 1

# College Dataset
# Problems: compare institutions, predict completion success, identify factors tied to graduation outcomes
# Main question: What characteristics predict whether a college has a high graduation rate?

# Job Placement Dataset
# Problems: predict employment outcome, identify at-risk groups, see which academic features matter most
# Main question: Which factors contribute most to job placement?

# %%

# Step 2: Data Preparation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# JOB PLACEMENT DATASET
# Main question: Which factors contribute most to job placement?
# IBM (Independent Business Metric): Placement rate

# Load data
placement_df = pd.read_csv(
    "https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv"
)

# Drop unneeded variables (sl_no is ID, salary is leakage)
placement_df = placement_df.drop(columns=["sl_no", "salary"])

# Create target variable (1 = Placed, 0 = Not Placed)
placement_df["target"] = (placement_df["status"] == "Placed").astype(int)
placement_df = placement_df.drop(columns=["status"])

# Calculate prevalence
print(f"[Placement] Target prevalence = {placement_df['target'].mean():.2%}")

# Split features and target
X_p = placement_df.drop(columns=["target"])
y_p = placement_df["target"]

# Create Train/Tune/Test partitions (60/20/20), stratified
X_train_p, X_temp_p, y_train_p, y_temp_p = train_test_split(
    X_p, y_p, test_size=0.40, random_state=42, stratify=y_p
)
X_tune_p, X_test_p, y_tune_p, y_test_p = train_test_split(
    X_temp_p, y_temp_p, test_size=0.50, random_state=42, stratify=y_temp_p
)

# Define column types
placement_cat = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
placement_num = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

# Correct variable types - categorical
for col in placement_cat:
    X_train_p[col] = X_train_p[col].astype("category")
    X_tune_p[col] = X_tune_p[col].astype("category")
    X_test_p[col] = X_test_p[col].astype("category")

# Correct variable types - numeric
for col in placement_num:
    X_train_p[col] = pd.to_numeric(X_train_p[col], errors="coerce")
    X_tune_p[col] = pd.to_numeric(X_tune_p[col], errors="coerce")
    X_test_p[col] = pd.to_numeric(X_test_p[col], errors="coerce")

# Fill missing numeric values using TRAIN medians (no data leakage)
for col in placement_num:
    med = X_train_p[col].median()
    X_train_p[col] = X_train_p[col].fillna(med)
    X_tune_p[col] = X_tune_p[col].fillna(med)
    X_test_p[col] = X_test_p[col].fillna(med)

# One-hot encode categorical variables (fit on train, align tune/test)
# No collapse needed - all categories have sufficient counts
X_train_p = pd.get_dummies(X_train_p, drop_first=True)
X_tune_p = pd.get_dummies(X_tune_p, drop_first=True).reindex(columns=X_train_p.columns, fill_value=0)
X_test_p = pd.get_dummies(X_test_p, drop_first=True).reindex(columns=X_train_p.columns, fill_value=0)

# Normalize continuous variables (fit on train only)
scaler_p = MinMaxScaler()
X_train_p[placement_num] = scaler_p.fit_transform(X_train_p[placement_num])
X_tune_p[placement_num] = scaler_p.transform(X_tune_p[placement_num])
X_test_p[placement_num] = scaler_p.transform(X_test_p[placement_num])

print("[Placement] Shapes:", X_train_p.shape, X_tune_p.shape, X_test_p.shape)


# Main question: What characteristics predict whether a college has a high graduation rate?
# IBM (Independent Business Metric): Graduation rate

# Load data
college_df = pd.read_csv(
    "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
)

# Keep only columns needed (drop unneeded variables)
keep_cols = [
    "level", "control", "hbcu", "flagship",
    "student_count", "ft_pct", "pell_value", "retain_value",
    "grad_150_value"
]
college_df = college_df[keep_cols].copy()

# Correct types: hbcu/flagship are "X" or blank -> convert to 1/0
college_df["hbcu"] = (college_df["hbcu"].fillna("").astype(str).str.upper() == "X").astype(int)
college_df["flagship"] = (college_df["flagship"].fillna("").astype(str).str.upper() == "X").astype(int)

# Drop rows missing the target
college_df = college_df.dropna(subset=["grad_150_value"])

# Create target variable: "high graduation" = 1 if grad_150_value >= median
median_grad = college_df["grad_150_value"].median()
college_df["target"] = (college_df["grad_150_value"] >= median_grad).astype(int)

# Drop original target column to prevent leakage
college_df = college_df.drop(columns=["grad_150_value"])

# Calculate prevalence
print(f"\n[College] Target prevalence = {college_df['target'].mean():.2%}")

# Split features and target
X_c = college_df.drop(columns=["target"])
y_c = college_df["target"]

# Create Train/Tune/Test partitions (60/20/20), stratified
X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X_c, y_c, test_size=0.40, random_state=42, stratify=y_c
)
X_tune_c, X_test_c, y_tune_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=0.50, random_state=42, stratify=y_temp_c
)

# Correct variable types - categorical
for col in ["level", "control"]:
    X_train_c[col] = X_train_c[col].astype("category")
    X_tune_c[col] = X_tune_c[col].astype("category")
    X_test_c[col] = X_test_c[col].astype("category")

# Correct variable types - numeric
college_num = ["hbcu", "flagship", "student_count", "ft_pct", "pell_value", "retain_value"]
for col in college_num:
    X_train_c[col] = pd.to_numeric(X_train_c[col], errors="coerce")
    X_tune_c[col] = pd.to_numeric(X_tune_c[col], errors="coerce")
    X_test_c[col] = pd.to_numeric(X_test_c[col], errors="coerce")

# Fill missing numeric values using TRAIN medians (no data leakage)
for col in college_num:
    med = X_train_c[col].median()
    X_train_c[col] = X_train_c[col].fillna(med)
    X_tune_c[col] = X_tune_c[col].fillna(med)
    X_test_c[col] = X_test_c[col].fillna(med)

# One-hot encode categorical variables (fit on train, align tune/test)
# No collapse needed - level and control have sufficient counts
X_train_c = pd.get_dummies(X_train_c, drop_first=True)
X_tune_c = pd.get_dummies(X_tune_c, drop_first=True).reindex(columns=X_train_c.columns, fill_value=0)
X_test_c = pd.get_dummies(X_test_c, drop_first=True).reindex(columns=X_train_c.columns, fill_value=0)

# Normalize continuous variables (fit on train only)
scaler_c = MinMaxScaler()
X_train_c[college_num] = scaler_c.fit_transform(X_train_c[college_num])
X_tune_c[college_num] = scaler_c.transform(X_tune_c[college_num])
X_test_c[college_num] = scaler_c.transform(X_test_c[college_num])

print("[College] Shapes:", X_train_c.shape, X_tune_c.shape, X_test_c.shape)

# %%

# Step 3: Instincts and Concerns

# COLLEGE DATASET
# Instincts: Good for finding patterns across institutions linked to higher graduation outcomes.
# Worries: It's institution-level data (not student-level), so associations aren't necessarily causal.
# Missing data could also bias results, and the "high" label depends on the median cutoff.

# JOB PLACEMENT DATASET
# Instincts: Good for predicting placement from academics, work experience, and specialization.
# Worries: Might be from a specific context so it may not generalize. Some predictors reflect structural factors.
# %%
