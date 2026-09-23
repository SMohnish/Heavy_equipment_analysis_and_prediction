# Heavy Equipment Selling Price Prediction

Predicts resale price of heavy equipment (excavators, wheel loaders, motorgraders etc.) from transaction, spec and usage data. Metric is RMSLE since the target ranges from a few thousand to a few hundred thousand dollars and we care about relative error, not absolute.

## Structure

```
heavy-equipment-price-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py            # paths, constants, seeds, search params
│   ├── preprocessing.py     # load, clean, feature engg, target encoding
│   ├── model_building.py    # baselines, tuning, CV, ensembling
│   ├── results.py           # rmsle, comparison table/plots, submission
│   └── train.py             # runs the full pipeline
└── outputs/                 # plots + submission.csv (created on run)
```

## What's happening in each file

**preprocessing.py**

Load train/test with explicit dtypes (numeric cols vs everything else as str), parse TransactionDate separately. ManufactureYear has junk values below 1900 in it so those get set to NaN before computing age.

Feature engineering is mostly mechanical stuff built off age and hours:
- `MachineAge` = TransactionYear - ManufactureYear
- `HoursPerYear`, `LogHours`, `LogAge` and an age*hours interaction term
- bucketed versions of age and hours (pd.cut) since the boosters can pick up thresholds on their own but bucketing still helped a bit
- `FunctionalClassification` is a text field like "Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons" — wrote a parser to split it into EquipType, SizeMid (midpoint of the range) and SizeUnit
- ordinal encode UtilizationTier and AssetScaleFactor since they're actually ordered (Low/Medium/High etc), not just categories
- a few interaction keys — ProductConfigID x Region, ProductConfigID x Year, VendorPartnerID x EquipType — since price for the same model can shift by region/year/dealer

For categoricals: anything with <=15 unique values gets one-hot encoded, anything above that gets K-fold target encoded (mean target per category, smoothed toward global mean) so we don't leak. Test set encoding uses stats from the entire train set since there's no leakage risk there.

**model_building.py**

Linear Regression and Random Forest as baselines on a single 80/20 split, just to sanity check the boosters are actually needed (they are — non-linear age/hours relationships aren't linear-model friendly).

RandomizedSearchCV for XGBoost and LightGBM, run on a 35% sample of the data to keep it fast (50 iters for xgb, 25 for lgb, 3-fold).

Final models — xgb, lgb, catboost — trained with 5-fold CV, repeated across multiple seeds each and averaged. Catboost params are just derived from the tuned xgb params (depth capped at 10 since catboost trees are symmetric, l2 reg doubled) instead of tuning it separately.

For the final ensemble, tried both a Ridge meta-model on top of the 3 models' OOF preds, and a plain grid-searched weighted average, and just picked whichever gave lower RMSLE on OOF.

**results.py**

rmsle_from_log_preds converts back from log1p scale before scoring. Also has the model comparison bar plot, feature importance plot (top 20, from the tuned xgb model) and the submission.csv writer.

**train.py**

Wires all of the above together end to end.

## Setup

```bash
pip install -r requirements.txt
```

Put train.csv and test.csv in a `data/` folder at repo root (or set DATA_DIR env var).

## Running

```bash
python src/train.py
```

Writes submission.csv, model_comparison.png and feature_importance.png to `outputs/`.

Or just import pieces:

```python
from preprocessing import load_raw_data, engineer_features, build_model_matrices

train, test = load_raw_data()
train_fe, test_fe = engineer_features(train), engineer_features(test)
X, X_test, y, y_raw, feature_names, preprocessor = build_model_matrices(train_fe, test_fe)
```

## Notes / things I noticed while building this

- OperationalHoursMeter has basically zero linear correlation with price but clearly matters when you look at it non-linearly (scatter plots showed it), which is why linear regression underperforms so much here.
- Price variance within the same ProductConfigID is way lower than overall price variance — that's basically why target encoding it works so well.
- Ended up below the ~0.20 RMSLE with the ensemble.
