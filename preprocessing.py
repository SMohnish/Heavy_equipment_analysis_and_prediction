import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import config


def build_dtype_map(cols):
    dtype_map = {}
    for c in cols:
        if c == "TransactionDate":
            continue
        if c in config.numeric_cols_raw:
            dtype_map[c] = "float64"
        else:
            dtype_map[c] = "str"
    return dtype_map


def load_raw_data(train_path=None, test_path=None):
    train_path = train_path or config.TRAIN_PATH
    test_path = test_path or config.TEST_PATH

    train_header = pd.read_csv(train_path, nrows=0).columns.tolist()
    test_header = pd.read_csv(test_path, nrows=0).columns.tolist()

    train = pd.read_csv(train_path, dtype=build_dtype_map(train_header), parse_dates=["TransactionDate"])
    test = pd.read_csv(test_path, dtype=build_dtype_map(test_header), parse_dates=["TransactionDate"])

    for c in config.id_and_year_cols:
        train[c] = train[c].astype(int)
        if c in test.columns:
            test[c] = test[c].astype(int)

    return train, test


def parse_classification(text_value):
    if pd.isna(text_value):
        return pd.Series([np.nan, np.nan, np.nan])
    text_value = str(text_value)
    if " - " not in text_value:
        return pd.Series([text_value, np.nan, np.nan])

    equip_type, size_part = text_value.split(" - ", 1)
    equip_type = equip_type.strip()
    size_part = size_part.strip()

    if " + " in size_part:
        number_text, unit_text = size_part.split(" + ", 1)
        try:
            return pd.Series([equip_type, float(number_text.strip()), unit_text.strip()])
        except ValueError:
            return pd.Series([equip_type, np.nan, np.nan])

    if " to " in size_part:
        try:
            low_text, remainder = size_part.split(" to ", 1)
            remainder_parts = remainder.strip().split(" ", 1)
            high_text = remainder_parts[0]
            unit_text = remainder_parts[1] if len(remainder_parts) > 1 else ""
            low_value, high_value = float(low_text.strip()), float(high_text.strip())
            size_midpoint = (low_value + high_value) / 2
            return pd.Series([equip_type, size_midpoint, unit_text.strip()])
        except (ValueError, IndexError):
            return pd.Series([equip_type, np.nan, np.nan])

    return pd.Series([equip_type, np.nan, np.nan])


def engineer_features(df):
    df = df.copy()

    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionDayOfYear"] = df["TransactionDate"].dt.dayofyear

    # < 1900 is a placeholder, not a real year
    df.loc[df["ManufactureYear"] < 1900, "ManufactureYear"] = np.nan
    df["MachineAge"] = df["TransactionYear"] - df["ManufactureYear"]
    df.loc[df["MachineAge"] < 0, "MachineAge"] = np.nan

    df["HoursMissingOrZero"] = ((df["OperationalHoursMeter"].isna()) | (df["OperationalHoursMeter"] == 0)).astype(int)
    age_filled = df["MachineAge"].fillna(df["MachineAge"].median())
    df["HoursPerYear"] = df["OperationalHoursMeter"].fillna(0) / (age_filled + 1)
    df["LogHours"] = np.log1p(df["OperationalHoursMeter"].fillna(0))
    df["LogAge"] = np.log1p(age_filled)
    df["Age_x_HoursPerYear"] = age_filled * df["HoursPerYear"]

    df["AgeBucket"] = pd.cut(
        df["MachineAge"], bins=[-1, 2, 5, 10, 15, 25, 100],
        labels=["0-2", "3-5", "6-10", "11-15", "16-25", "25+"]
    ).astype(str)
    df["HoursBucket"] = pd.cut(
        df["OperationalHoursMeter"], bins=[-1, 0, 1000, 3000, 6000, 12000, 1e9],
        labels=["0", "1-1000", "1000-3000", "3000-6000", "6000-12000", "12000+"]
    ).astype(str)

    df["IsLowUtilizationOldMachine"] = (
        (df["MachineAge"] > 10) & (df["HoursPerYear"] < df["HoursPerYear"].median())
    ).astype(int)
    df["BaseClass_AgeBucket"] = df["Spec_BaseClass"].astype(str) + "_" + df["AgeBucket"]

    df["UtilizationTier_ord"] = df["UtilizationTier"].map(config.utilization_order)
    df["AssetScaleFactor_ord"] = df["AssetScaleFactor"].map(config.scale_order)

    df["ProductConfigID"] = df["ProductConfigID"].astype(str)

    parsed = df["FunctionalClassification"].apply(parse_classification)
    parsed.columns = ["EquipType", "SizeMid", "SizeUnit"]
    df["EquipType"] = parsed["EquipType"]
    df["SizeMid"] = parsed["SizeMid"]
    df["SizeUnit"] = parsed["SizeUnit"]

    df["PCID_x_Region"] = df["ProductConfigID"].astype(str) + "_" + df["RegionCode"].astype(str)
    df["PCID_x_Year"] = df["ProductConfigID"].astype(str) + "_" + df["TransactionYear"].astype(str)
    df["Vendor_x_EquipType"] = df["VendorPartnerID"].astype(str) + "_" + df["EquipType"].astype(str)

    df = df.drop(columns=[c for c in config.columns_to_drop if c in df.columns])
    return df


def kfold_target_encode(train_df, test_df, column_name, target_column,
                         n_splits=None, smoothing=None):
    n_splits = n_splits or config.TARGET_ENCODE_N_SPLITS
    smoothing = smoothing or config.TARGET_ENCODE_SMOOTHING

    overall_mean = train_df[target_column].mean()
    train_encoded_values = pd.Series(index=train_df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for train_idx, holdout_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        fold_holdout = train_df.iloc[holdout_idx]

        category_stats = fold_train.groupby(column_name)[target_column].agg(["mean", "count"])
        smoothed_mean = (
            (category_stats["mean"] * category_stats["count"] + overall_mean * smoothing)
            / (category_stats["count"] + smoothing)
        )
        train_encoded_values.iloc[holdout_idx] = fold_holdout[column_name].map(smoothed_mean).fillna(overall_mean).values

    # test set uses stats from the full training set, no leakage risk here
    full_stats = train_df.groupby(column_name)[target_column].agg(["mean", "count"])
    full_smoothed_mean = (
        (full_stats["mean"] * full_stats["count"] + overall_mean * smoothing)
        / (full_stats["count"] + smoothing)
    )
    test_encoded_values = test_df[column_name].map(full_smoothed_mean).fillna(overall_mean)

    return train_encoded_values.values, test_encoded_values.values


def apply_target_encoding(train_fe, test_fe, high_cardinality_cols, target_col=None):
    target_col = target_col or config.TARGET_COL
    target_encoded_cols = []
    for c in high_cardinality_cols:
        encoded_col_name = c + "_te"
        train_fe[encoded_col_name], test_fe[encoded_col_name] = kfold_target_encode(
            train_fe, test_fe, c, target_col
        )
        target_encoded_cols.append(encoded_col_name)

    train_fe = train_fe.drop(columns=high_cardinality_cols)
    test_fe = test_fe.drop(columns=high_cardinality_cols)
    return train_fe, test_fe, target_encoded_cols


def classify_columns(train_fe, target_col=None, raw_target_col=None):
    target_col = target_col or config.TARGET_COL
    raw_target_col = raw_target_col or config.RAW_TARGET_COL

    feature_cols = [c for c in train_fe.columns if c not in [target_col, raw_target_col]]
    categorical_cols = [c for c in feature_cols if train_fe[c].dtype == object or str(train_fe[c].dtype) == "str"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    category_counts = train_fe[categorical_cols].nunique().sort_values(ascending=False)
    low_cardinality_cols = category_counts[category_counts <= config.HIGH_CARDINALITY_THRESHOLD].index.tolist()
    high_cardinality_cols = category_counts[category_counts > config.HIGH_CARDINALITY_THRESHOLD].index.tolist()

    return numeric_cols, categorical_cols, low_cardinality_cols, high_cardinality_cols


def build_preprocessor(numeric_features, low_cardinality_cols):
    numeric_pipeline = Pipeline([
        ("impute_missing_values", SimpleImputer(strategy="median")),
        ("scale_to_similar_range", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute_missing_values", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, low_cardinality_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_matrices(train_fe, test_fe):
    train_fe = train_fe.copy()
    test_fe = test_fe.copy()
    train_fe[config.TARGET_COL] = np.log1p(train_fe[config.RAW_TARGET_COL])

    numeric_cols, categorical_cols, low_cardinality_cols, high_cardinality_cols = classify_columns(train_fe)

    for c in categorical_cols:
        train_fe[c] = train_fe[c].fillna("Missing")
        test_fe[c] = test_fe[c].fillna("Missing")

    train_fe, test_fe, target_encoded_cols = apply_target_encoding(train_fe, test_fe, high_cardinality_cols)

    numeric_features = numeric_cols + target_encoded_cols
    preprocessor = build_preprocessor(numeric_features, low_cardinality_cols)

    all_feature_cols = numeric_features + low_cardinality_cols
    X = preprocessor.fit_transform(train_fe[all_feature_cols])
    X_test = preprocessor.transform(test_fe[all_feature_cols])

    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y = train_fe[config.TARGET_COL].values
    y_raw = train_fe[config.RAW_TARGET_COL].values
    feature_names = preprocessor.get_feature_names_out()

    return X, X_test, y, y_raw, feature_names, preprocessor
