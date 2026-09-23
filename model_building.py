import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import config
from results import rmsle_from_log_preds


def make_train_val_split(X, y, y_raw, test_size=0.2):
    return train_test_split(X, y, y_raw, test_size=test_size, random_state=config.RANDOM_STATE)


def train_baselines(X_train, y_train, X_val, y_val_raw):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_val)
    print("Linear Regression RMSLE:", rmsle_from_log_preds(y_val_raw, linear_preds))

    random_forest_model = RandomForestRegressor(
        n_estimators=300, max_depth=16, min_samples_leaf=2,
        max_features=0.6, n_jobs=4, random_state=config.RANDOM_STATE,
    )
    random_forest_model.fit(X_train, y_train)
    random_forest_preds = random_forest_model.predict(X_val)
    print("Random Forest RMSLE:", rmsle_from_log_preds(y_val_raw, random_forest_preds))

    return linear_model, linear_preds, random_forest_model, random_forest_preds


def tune_xgboost(X, y):
    search_sample_index = pd.Series(range(len(X))).sample(
        frac=config.SEARCH_SAMPLE_FRAC, random_state=config.RANDOM_STATE
    ).index
    X_search, y_search = X[search_sample_index], y[search_sample_index]

    xgb_param_dist = {
        "max_depth": [4, 5, 6, 7, 8, 9, 10, 11],
        "learning_rate": [0.008, 0.01, 0.015, 0.02, 0.025, 0.03],
        "n_estimators": [1000, 1500, 2000, 2500],
        "subsample": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 2, 3, 5, 7, 10],
        "reg_lambda": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "reg_alpha": [0, 0.05, 0.1, 0.2, 0.5, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0, 1.2],
    }

    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(n_jobs=4, random_state=config.RANDOM_STATE),
        param_distributions=xgb_param_dist,
        n_iter=config.XGB_SEARCH_N_ITER,
        cv=config.SEARCH_CV_FOLDS,
        scoring="neg_mean_squared_error",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    xgb_search.fit(X_search, y_search)
    print("Best XGBoost params:", xgb_search.best_params_)
    return xgb_search.best_params_


def tune_lightgbm(X, y):
    search_sample_index = pd.Series(range(len(X))).sample(
        frac=config.SEARCH_SAMPLE_FRAC, random_state=config.RANDOM_STATE
    ).index
    X_search, y_search = X[search_sample_index], y[search_sample_index]

    lgb_param_dist = {
        "max_depth": [5, 6, 7, 8, 9, -1],
        "num_leaves": [31, 47, 63, 95, 127],
        "learning_rate": [0.01, 0.015, 0.02, 0.03],
        "n_estimators": [1000, 1500, 2000],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_samples": [5, 10, 15, 20, 30],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(n_jobs=4, random_state=config.RANDOM_STATE, verbose=-1, feature_name=False),
        param_distributions=lgb_param_dist,
        n_iter=config.LGB_SEARCH_N_ITER,
        cv=config.SEARCH_CV_FOLDS,
        scoring="neg_mean_squared_error",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    lgb_search.fit(X_search, y_search)
    print("Best LightGBM params:", lgb_search.best_params_)
    return lgb_search.best_params_


def build_xgb_model(seed, best_xgb_params):
    return xgb.XGBRegressor(
        **best_xgb_params, n_jobs=4, random_state=seed,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, eval_metric="rmse",
    )


def build_lgb_model(seed, best_lgb_params):
    return lgb.LGBMRegressor(**best_lgb_params, n_jobs=4, random_state=seed, verbose=-1)


def build_catboost_model(seed, best_xgb_params):
    # reuse xgb's tuned params as a starting point, cap depth at 10, double the l2 reg
    return CatBoostRegressor(
        iterations=best_xgb_params["n_estimators"],
        learning_rate=best_xgb_params["learning_rate"],
        depth=min(best_xgb_params["max_depth"], 10),
        l2_leaf_reg=best_xgb_params["reg_lambda"] * 2,
        random_state=seed, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose=False, thread_count=4,
    )


def run_cross_validation(X, y, y_raw, X_test, build_model, model_name, n_splits=None):
    n_splits = n_splits or config.CV_N_SPLITS
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    out_of_fold_preds = np.zeros(len(X))
    test_fold_preds = []

    for fold_number, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        model = build_model()

        if model_name.startswith("xgb"):
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        elif model_name.startswith("lgb"):
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)],
                      callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
        elif model_name.startswith("cb"):
            model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), verbose=False)

        out_of_fold_preds[val_idx] = model.predict(X_fold_val)
        test_fold_preds.append(model.predict(X_test))

        fold_score = rmsle_from_log_preds(y_raw[val_idx], out_of_fold_preds[val_idx])
        print(f"{model_name} fold {fold_number}: RMSLE = {fold_score:.5f}", flush=True)

    overall_score = rmsle_from_log_preds(y_raw, out_of_fold_preds)
    print(f"{model_name} OOF RMSLE: {overall_score:.5f}\n", flush=True)
    return out_of_fold_preds, np.mean(test_fold_preds, axis=0)


def run_cross_validation_multi_seed(X, y, y_raw, X_test, build_model_for_seed, model_name, seeds):
    seed_oof_preds, seed_test_preds = [], []
    for seed in seeds:
        oof, test_p = run_cross_validation(
            X, y, y_raw, X_test, lambda s=seed: build_model_for_seed(s), f"{model_name}_seed{seed}"
        )
        seed_oof_preds.append(oof)
        seed_test_preds.append(test_p)
    return np.mean(seed_oof_preds, axis=0), np.mean(seed_test_preds, axis=0)


def ensemble_predictions(oof_xgb, oof_lgb, oof_cb, test_xgb, test_lgb, test_cb, y, y_raw):
    stacked_features = np.column_stack([oof_xgb, oof_lgb, oof_cb])
    stacked_test_features = np.column_stack([test_xgb, test_lgb, test_cb])

    best_ridge_score, best_ridge_alpha, best_ridge_model = 999, None, None
    for alpha in [1.0, 3.0, 5.0, 10.0, 20.0]:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(stacked_features, y)
        ridge_preds = ridge_model.predict(stacked_features)
        score = rmsle_from_log_preds(y_raw, ridge_preds)
        print(f"Ridge alpha={alpha}: RMSLE={score:.5f}, coefficients={ridge_model.coef_}")
        if score < best_ridge_score:
            best_ridge_score, best_ridge_alpha, best_ridge_model = score, alpha, ridge_model

    best_blend_score, best_weights = 999, None
    weight_options = np.arange(0, 1.05, 0.05)
    for w_xgb in weight_options:
        for w_lgb in weight_options:
            for w_cb in weight_options:
                total_weight = w_xgb + w_lgb + w_cb
                if total_weight == 0:
                    continue
                blended_preds = (w_xgb * oof_xgb + w_lgb * oof_lgb + w_cb * oof_cb) / total_weight
                score = rmsle_from_log_preds(y_raw, blended_preds)
                if score < best_blend_score:
                    best_blend_score = score
                    best_weights = (w_xgb / total_weight, w_lgb / total_weight, w_cb / total_weight)

    print(f"Best Ridge:              alpha={best_ridge_alpha}, RMSLE={best_ridge_score:.5f}")
    print(f"Best constrained blend:  weights={best_weights}, RMSLE={best_blend_score:.5f}")

    use_ridge = best_ridge_score <= best_blend_score
    print("\nUsing for final submission:", "Ridge" if use_ridge else "constrained weighted average")

    if use_ridge:
        final_log_preds = best_ridge_model.predict(stacked_test_features)
    else:
        w_xgb, w_lgb, w_cb = best_weights
        final_log_preds = w_xgb * test_xgb + w_lgb * test_lgb + w_cb * test_cb

    summary = {
        "best_ridge_alpha": best_ridge_alpha,
        "best_ridge_rmsle": best_ridge_score,
        "best_blend_weights": best_weights,
        "best_blend_rmsle": best_blend_score,
        "used_method": "ridge" if use_ridge else "weighted_average",
    }
    return final_log_preds, summary
