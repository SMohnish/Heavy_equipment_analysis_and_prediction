import xgboost as xgb

import config
from preprocessing import load_raw_data, engineer_features, build_model_matrices
from model_building import (
    make_train_val_split,
    train_baselines,
    tune_xgboost,
    tune_lightgbm,
    build_xgb_model,
    build_lgb_model,
    build_catboost_model,
    run_cross_validation_multi_seed,
    ensemble_predictions,
)
from results import build_comparison_table, plot_model_comparison, plot_feature_importance, build_submission


def main():
    print("Loading raw data...")
    train, test = load_raw_data()
    print("Train shape:", train.shape, "| Test shape:", test.shape)

    print("\nEngineering features...")
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    print("\nBuilding model matrices...")
    X, X_test, y, y_raw, feature_names, preprocessor = build_model_matrices(train_fe, test_fe)
    print("X shape:", X.shape, "| X_test shape:", X_test.shape)

    print("\nTraining baseline models...")
    X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = make_train_val_split(X, y, y_raw)
    linear_model, linear_preds, random_forest_model, random_forest_preds = train_baselines(
        X_train, y_train, X_val, y_val_raw
    )

    print("\nTuning XGBoost...")
    best_xgb_params = tune_xgboost(X, y)

    print("\nTuning LightGBM...")
    best_lgb_params = tune_lightgbm(X, y)

    print("\nRunning XGBoost multi-seed cross-validation...")
    oof_xgb, test_xgb = run_cross_validation_multi_seed(
        X, y, y_raw, X_test,
        lambda seed: build_xgb_model(seed, best_xgb_params), "xgb", seeds=config.XGB_SEEDS,
    )

    print("\nRunning LightGBM multi-seed cross-validation...")
    oof_lgb, test_lgb = run_cross_validation_multi_seed(
        X, y, y_raw, X_test,
        lambda seed: build_lgb_model(seed, best_lgb_params), "lgb", seeds=config.LGB_SEEDS,
    )

    print("\nRunning CatBoost multi-seed cross-validation...")
    oof_cb, test_cb = run_cross_validation_multi_seed(
        X, y, y_raw, X_test,
        lambda seed: build_catboost_model(seed, best_xgb_params), "cb", seeds=config.CB_SEEDS,
    )

    print("\nBuilding model comparison...")
    comparison_df = build_comparison_table(
        y_val_raw, linear_preds, random_forest_preds, y_raw, oof_xgb, oof_lgb, oof_cb
    )
    print(comparison_df)
    plot_model_comparison(comparison_df)

    print("\nFitting model for feature importance...")
    importance_model = xgb.XGBRegressor(**best_xgb_params, n_jobs=4, random_state=config.RANDOM_STATE)
    importance_model.fit(X_train, y_train)
    plot_feature_importance(importance_model, feature_names)

    print("\nEnsembling model predictions...")
    final_log_preds, ensemble_summary = ensemble_predictions(
        oof_xgb, oof_lgb, oof_cb, test_xgb, test_lgb, test_cb, y, y_raw
    )
    print("Ensemble summary:", ensemble_summary)

    print("\nBuilding submission file...")
    build_submission(test["TransactionID"], final_log_preds)


if __name__ == "__main__":
    main()
