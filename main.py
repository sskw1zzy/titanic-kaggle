import re
import pandas as pd

from config import SEED, AGE_BINS, FARE_QCUT_Q, COMMON_TITLES, SCALING_COLS, DROP_COLS, ONE_HOT_COLS
from src.data import load_data
from src.features import add_has_cabin, add_title, add_bins, one_hot_encoding, drop_unused, add_family, add_scaling
from src.models import get_forest_model


def preprocess(df, fare_bins=None, scaler=None):
    """Full Titanic feature engineering pipeline. Pass fare_bins/scaler from
    the train call when processing test."""
    df = add_has_cabin(df)
    df = add_title(df, COMMON_TITLES)
    df, fare_bins = add_bins(df, AGE_BINS, FARE_QCUT_Q, fare_bins=fare_bins)
    df = one_hot_encoding(df, ONE_HOT_COLS)
    df = drop_unused(df, DROP_COLS)
    df = add_family(df)
    df, scaler = add_scaling(df, SCALING_COLS, scaler=scaler)
    return df, fare_bins, scaler


def clean_column_names(df):
    """Strips characters not accepted by XGBoost/LightGBM."""
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in df.columns]
    return df


train_df, test_df = load_data()

train_df, fare_bins, scaler = preprocess(train_df)
X = clean_column_names(train_df.drop(columns=['Survived', 'PassengerId']))
y = train_df['Survived']

test_df, _, _ = preprocess(test_df, fare_bins=fare_bins, scaler=scaler)
test_ids = test_df['PassengerId']
X_test = clean_column_names(test_df.drop(columns=['PassengerId']))

# Best params for Random Forest after tuning. Full comparison in RESULTS.md.
model = get_forest_model(SEED, n_estimators=50, max_depth=5, min_samples_leaf=1)
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
submission.to_csv('submission.csv', index=False)