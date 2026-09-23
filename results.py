import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error

import config


def rmsle_from_log_preds(true_prices, predicted_log_prices):
    predicted_prices = np.maximum(np.expm1(predicted_log_prices), 0)
    return np.sqrt(mean_squared_log_error(true_prices, predicted_prices))


def build_comparison_table(y_val_raw, linear_preds, random_forest_preds,
                            y_raw, oof_xgb, oof_lgb, oof_cb):
    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost (tuned, multi-seed)",
                  "LightGBM (tuned, multi-seed)", "CatBoost (multi-seed)"],
        "RMSLE": [
            rmsle_from_log_preds(y_val_raw, linear_preds),
            rmsle_from_log_preds(y_val_raw, random_forest_preds),
            rmsle_from_log_preds(y_raw, oof_xgb),
            rmsle_from_log_preds(y_raw, oof_lgb),
            rmsle_from_log_preds(y_raw, oof_cb),
        ],
    }).sort_values("RMSLE")
    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=comparison_df, x="RMSLE", y="Model", hue="Model", legend=False, palette="viridis", ax=ax)
    ax.axvline(0.20, color="red", linestyle="--", label="Competition cutoff (0.20)")
    ax.legend()
    ax.set_title("RMSLE by model (Linear/RF: single validation split, boosters: multi-seed 5-fold OOF)")
    plt.tight_layout()

    save_path = save_path or config.MODEL_COMPARISON_PLOT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved model comparison plot to {save_path}")


def plot_feature_importance(importance_model, feature_names, top_n=20, save_path=None):
    importances = pd.Series(importance_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, ax=ax, palette="rocket")
    ax.set_title(f"Top {top_n} feature importances (XGBoost, tuned)")
    plt.tight_layout()

    save_path = save_path or config.FEATURE_IMPORTANCE_PLOT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {save_path}")

    return importances


def build_submission(test_ids, final_log_preds, save_path=None):
    final_price_preds = np.maximum(np.expm1(final_log_preds), 0)

    submission = pd.DataFrame({
        "TransactionID": test_ids,
        "TargetValue": final_price_preds,
    })

    save_path = save_path or config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)
    print(f"Saved submission ({submission.shape}) to {save_path}")
    return submission
