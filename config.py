import os

DATA_DIR = os.environ.get("DATA_DIR", "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
MODEL_COMPARISON_PLOT = os.path.join(OUTPUT_DIR, "model_comparison.png")
FEATURE_IMPORTANCE_PLOT = os.path.join(OUTPUT_DIR, "feature_importance.png")

RANDOM_STATE = 42

numeric_cols_raw = ["TransactionID", "TargetValue", "AssetID", "ProductConfigID",
                    "ManufactureYear", "OperationalHoursMeter"]

id_and_year_cols = ["TransactionID", "AssetID", "ProductConfigID", "ManufactureYear"]

TARGET_COL = "LogTarget"
RAW_TARGET_COL = "TargetValue"

# columns that don't generalize or got replaced by engineered versions
columns_to_drop = ["TransactionID", "AssetID", "TransactionDate",
                    "UtilizationTier", "AssetScaleFactor", "col18", "col19"]

utilization_order = {"Low": 0, "Medium": 1, "High": 2}
scale_order = {"Mini": 0, "Compact": 1, "Small": 2, "Medium": 3, "Large / Medium": 4, "Large": 5}

HIGH_CARDINALITY_THRESHOLD = 15

TARGET_ENCODE_N_SPLITS = 10
TARGET_ENCODE_SMOOTHING = 3

CV_N_SPLITS = 5
XGB_SEEDS = (42, 7, 123, 2024, 99)
LGB_SEEDS = (42, 7, 123)
CB_SEEDS = (42, 7, 123)
EARLY_STOPPING_ROUNDS = 75

SEARCH_SAMPLE_FRAC = 0.35
XGB_SEARCH_N_ITER = 50
LGB_SEARCH_N_ITER = 25
SEARCH_CV_FOLDS = 3
