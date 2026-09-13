from sklearn.model_selection import  StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier


def evaluate_model(model, X, y, seed, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return scores

def evaluate_catboost(X, y, cat_features, seed, n_splits, **model_params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(random_state=seed, cat_features=cat_features, verbose=0, **model_params)
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    return scores