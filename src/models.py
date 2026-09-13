from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_baseline_model(seed):
    return LogisticRegression(random_state=seed, max_iter=1000)

def get_regularized_model(seed, penalty='l2', C=1.0, l1_ratio=None):
    return LogisticRegression(
        random_state=seed,
        max_iter=20000,
        penalty=penalty,
        C=C, 
        l1_ratio=l1_ratio,
        solver='saga'
    )

def get_knn_model(n_neighbors=5, weights='uniform', metric='minkowski'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

def get_tree_model(seed, max_depth=None, min_samples_leaf=1):
    return DecisionTreeClassifier(random_state=seed, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

def get_forest_model(seed, n_estimators=100, max_depth=None, min_samples_leaf=1):
    return RandomForestClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

def get_xgboost_model(seed, n_estimators=100, max_depth=6, learning_rate=0.1):
    return XGBClassifier(random_state=seed, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

def get_lightgbm_model(seed, n_estimators=100, max_depth=-1, learning_rate=0.1):
    return LGBMClassifier(random_state=seed, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, verbose=-1)

def get_catboost_model(seed, iterations=100, depth=8, learning_rate=0.1):
    return CatBoostClassifier(random_state=seed, iterations=iterations, depth=depth, learning_rate=learning_rate, verbose=0)