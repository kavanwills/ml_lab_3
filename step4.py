# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def placement_pipeline(random_state=42):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv"
    )

    df = df.drop(columns=["sl_no", "salary"])
    df["target"] = (df["status"] == "Placed").astype(int)
    df = df.drop(columns=["status"])

    prevalence = df["target"].mean()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
    num_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_tune[col] = X_tune[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_tune[col] = pd.to_numeric(X_tune[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_tune[col] = X_tune[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_tune = pd.get_dummies(X_tune, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_tune[num_cols] = scaler.transform(X_tune[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_tune, X_test, y_train, y_tune, y_test, prevalence


def college_pipeline(random_state=42):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
    )

    keep_cols = [
        "level", "control", "hbcu", "flagship",
        "student_count", "ft_pct", "pell_value", "retain_value",
        "grad_150_value"
    ]
    df = df[keep_cols].copy()

    df["hbcu"] = (df["hbcu"].fillna("").astype(str).str.upper() == "X").astype(int)
    df["flagship"] = (df["flagship"].fillna("").astype(str).str.upper() == "X").astype(int)

    df = df.dropna(subset=["grad_150_value"])
    threshold = df["grad_150_value"].median()
    df["target"] = (df["grad_150_value"] >= threshold).astype(int)
    df = df.drop(columns=["grad_150_value"])

    prevalence = df["target"].mean()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    cat_cols = ["level", "control"]
    num_cols = ["hbcu", "flagship", "student_count", "ft_pct", "pell_value", "retain_value"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_tune[col] = X_tune[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_tune[col] = pd.to_numeric(X_tune[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_tune[col] = X_tune[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_tune = pd.get_dummies(X_tune, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_tune[num_cols] = scaler.transform(X_tune[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_tune, X_test, y_train, y_tune, y_test, prevalence, threshold

# %%
