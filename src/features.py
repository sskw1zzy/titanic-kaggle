import pandas as pd
from sklearn.preprocessing import StandardScaler


def one_hot_encoding(df, columns):
    return pd.get_dummies(df, columns=columns)


def add_bins(df, age_bins, fare_q, fare_bins=None):
    """Buckets Age into fixed bins and Fare into quantile bins.

    Pass fare_bins from a prior train call to reuse the same edges for test.
    """
    df = df.copy()
    df['AgeBin'] = pd.cut(df['Age'], bins=age_bins)
    if fare_bins is None:
        df['FareBin'], fare_bins = pd.qcut(df['Fare'], q=fare_q, retbins=True)
    else:
        df['FareBin'] = pd.cut(df['Fare'], bins=fare_bins, include_lowest=True)
    return df, fare_bins


def add_title(df, common_titles):
    """Extracts a title from Name, keeps common_titles as-is, groups the rest as 'Rare'."""
    df = df.copy()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df.loc[~(df['Title'].isin(common_titles)), 'Title'] = 'Rare'
    return df


def add_has_cabin(df):
    """Replaces Cabin with a binary flag for whether it was known."""
    df = df.copy()
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    return df.drop(columns=['Cabin'])


def add_family(df):
    """Adds Family size (SibSp + Parch + 1) and an IsAlone flag."""
    df = df.copy()
    df['Family'] = df['Parch'] + df['SibSp'] + 1
    df['IsAlone'] = (df['Family'] == 1).astype(int)
    return df


def drop_unused(df, cols):
    return df.drop(columns=cols)


def add_scaling(df, numeric_cols, scaler=None):
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler