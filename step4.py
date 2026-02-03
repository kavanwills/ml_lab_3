# %%

# Step 4: Pipeline Functions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def placement_pipeline(random_state=42):
    """
    Complete data preparation pipeline for the Job Placement dataset.
    
    Steps:
        1. Load data
        2. Drop unneeded variables (ID, salary)
        3. Create binary target (Placed = 1)
        4. Split into Train/Tune/Test (60/20/20)
        5. Correct variable types
        6. Fill missing values with train medians
        7. One-hot encode categoricals
        8. Normalize continuous variables
    
    Parameters:
        random_state (int): Seed for reproducibility
    
    Returns:
        X_train, X_tune, X_test: Feature DataFrames
        y_train, y_tune, y_test: Target Series
        prevalence (float): Proportion of positive class
    """
    # Load data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv"
    )

    # Drop unneeded variables
    df = df.drop(columns=["sl_no", "salary"])
    
    # Create target variable
    df["target"] = (df["status"] == "Placed").astype(int)
    df = df.drop(columns=["status"])

    # Calculate prevalence
    prevalence = df["target"].mean()

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Create partitions (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Define column types
    cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
    num_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

    # Correct variable types - categorical
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_tune[col] = X_tune[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Correct variable types - numeric
    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_tune[col] = pd.to_numeric(X_tune[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill missing values with train medians
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_tune[col] = X_tune[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    # One-hot encode (fit on train, align tune/test)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_tune = pd.get_dummies(X_tune, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    # Normalize continuous variables (fit on train only)
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_tune[num_cols] = scaler.transform(X_tune[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_tune, X_test, y_train, y_tune, y_test, prevalence


def college_pipeline(random_state=42):
    """
    Complete data preparation pipeline for the College Completion dataset.
    
    Steps:
        1. Load data
        2. Keep only needed columns
        3. Convert hbcu/flagship flags to binary
        4. Create binary target (above median graduation rate)
        5. Split into Train/Tune/Test (60/20/20)
        6. Correct variable types
        7. Fill missing values with train medians
        8. One-hot encode categoricals
        9. Normalize continuous variables
    
    Parameters:
        random_state (int): Seed for reproducibility
    
    Returns:
        X_train, X_tune, X_test: Feature DataFrames
        y_train, y_tune, y_test: Target Series
        prevalence (float): Proportion of positive class
        threshold (float): Median graduation rate used for target
    """
    # Load data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
    )

    # Keep only needed columns
    keep_cols = [
        "level", "control", "hbcu", "flagship",
        "student_count", "ft_pct", "pell_value", "retain_value",
        "grad_150_value"
    ]
    df = df[keep_cols].copy()

    # Convert hbcu/flagship to binary (X -> 1, blank -> 0)
    df["hbcu"] = (df["hbcu"].fillna("").astype(str).str.upper() == "X").astype(int)
    df["flagship"] = (df["flagship"].fillna("").astype(str).str.upper() == "X").astype(int)

    # Drop rows missing target
    df = df.dropna(subset=["grad_150_value"])
    
    # Create target variable (above median = 1)
    threshold = df["grad_150_value"].median()
    df["target"] = (df["grad_150_value"] >= threshold).astype(int)
    df = df.drop(columns=["grad_150_value"])

    # Calculate prevalence
    prevalence = df["target"].mean()

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Create partitions (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Define column types
    cat_cols = ["level", "control"]
    num_cols = ["hbcu", "flagship", "student_count", "ft_pct", "pell_value", "retain_value"]

    # Correct variable types - categorical
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_tune[col] = X_tune[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Correct variable types - numeric
    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_tune[col] = pd.to_numeric(X_tune[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill missing values with train medians
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_tune[col] = X_tune[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    # One-hot encode (fit on train, align tune/test)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_tune = pd.get_dummies(X_tune, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    # Normalize continuous variables (fit on train only)
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_tune[num_cols] = scaler.transform(X_tune[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_tune, X_test, y_train, y_tune, y_test, prevalence, threshold

# %%
