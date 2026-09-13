import pandas as pd
from config import DATA_PATHS


def load_data():
    """Loads train and test CSVs from the paths configured in config.py."""
    train_df = pd.read_csv(DATA_PATHS['train'])
    test_df = pd.read_csv(DATA_PATHS['test'])
    return train_df, test_df