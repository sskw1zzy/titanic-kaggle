import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LinearRegression

from src.models import get_lightgbm_model, get_baseline_model, get_knn_model
from src.dnn import DNNModel, prepare_dnn_data, train_dnn


def get_dnn_probs(X_dnn_train, y_train, X_dnn_eval, seed, dnn_model_params, dnn_train_params):
    """Trains a DNN and returns predicted probabilities on X_dnn_eval."""
    torch.manual_seed(seed)
    train_loader = prepare_dnn_data(X_dnn_train, y_train, batch_size=dnn_train_params['batch_size'])
    model = DNNModel(input_size=X_dnn_train.shape[1], **dnn_model_params)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=dnn_train_params['lr'])
    model = train_dnn(model, train_loader, loss_fn, optimizer, dnn_train_params['num_epochs'])

    model.eval()
    with torch.no_grad():
        X_eval_tensor = torch.tensor(X_dnn_eval.astype('float32').values, dtype=torch.float32)
        probs = torch.sigmoid(model(X_eval_tensor)).squeeze().numpy()
    return probs


def evaluate_ensemble_averaging(X, y, X_dnn, seed, n_splits, lgbm_params, dnn_model_params, dnn_train_params, weights=(1, 1, 1)):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgbm_model = get_lightgbm_model(seed, **lgbm_params)
        lgbm_model.fit(X_train, y_train)
        lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

        logreg_model = get_baseline_model(seed)
        logreg_model.fit(X_train, y_train)
        logreg_probs = logreg_model.predict_proba(X_val)[:, 1]

        X_dnn_train, X_dnn_val = X_dnn.iloc[train_idx], X_dnn.iloc[val_idx]
        dnn_probs = get_dnn_probs(X_dnn_train, y_train, X_dnn_val, seed, dnn_model_params, dnn_train_params)

        w_lgbm, w_logreg, w_dnn = weights
        avg_probs = (lgbm_probs * w_lgbm + logreg_probs * w_logreg + dnn_probs * w_dnn) / sum(weights)
        predictions = (avg_probs > 0.5).astype(int)
        accuracy = (predictions == y_val.values).mean()
        scores.append(accuracy)

    return scores


def evaluate_ensemble_averaging_v2(X, y, X_dnn, seed, n_splits, lgbm_params, knn_params, dnn_model_params, dnn_train_params, weights=(1, 1, 1)):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgbm_model = get_lightgbm_model(seed, **lgbm_params)
        lgbm_model.fit(X_train, y_train)
        lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

        knn_model = get_knn_model(**knn_params)
        knn_model.fit(X_train, y_train)
        knn_probs = knn_model.predict_proba(X_val)[:, 1]

        X_dnn_train, X_dnn_val = X_dnn.iloc[train_idx], X_dnn.iloc[val_idx]
        dnn_probs = get_dnn_probs(X_dnn_train, y_train, X_dnn_val, seed, dnn_model_params, dnn_train_params)

        w_lgbm, w_knn, w_dnn = weights
        avg_probs = (lgbm_probs * w_lgbm + knn_probs * w_knn + dnn_probs * w_dnn) / sum(weights)
        predictions = (avg_probs > 0.5).astype(int)
        accuracy = (predictions == y_val.values).mean()
        scores.append(accuracy)

    return scores


def evaluate_ensemble_4models(X, X_ext, y, X_dnn, seed, n_splits, lgbm_params, knn_params, dnn_model_params, dnn_train_params, weights=(1, 1, 1, 1)):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        X_ext_train, X_ext_val = X_ext.iloc[train_idx], X_ext.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lgbm_model = get_lightgbm_model(seed, **lgbm_params)
        lgbm_model.fit(X_train, y_train)
        lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

        knn_model = get_knn_model(**knn_params)
        knn_model.fit(X_train, y_train)
        knn_probs = knn_model.predict_proba(X_val)[:, 1]

        logreg_model = get_baseline_model(seed)
        logreg_model.fit(X_ext_train, y_train)
        logreg_probs = logreg_model.predict_proba(X_ext_val)[:, 1]

        X_dnn_train, X_dnn_val = X_dnn.iloc[train_idx], X_dnn.iloc[val_idx]
        dnn_probs = get_dnn_probs(X_dnn_train, y_train, X_dnn_val, seed, dnn_model_params, dnn_train_params)

        w_lgbm, w_knn, w_logreg, w_dnn = weights
        avg_probs = (lgbm_probs * w_lgbm + knn_probs * w_knn + logreg_probs * w_logreg + dnn_probs * w_dnn) / sum(weights)
        predictions = (avg_probs > 0.5).astype(int)
        accuracy = (predictions == y_val.values).mean()
        scores.append(accuracy)

    return scores


def generate_oof_predictions(X, y, X_dnn, seed, n_splits, lgbm_params, knn_params, dnn_model_params, dnn_train_params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X), 3))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        lgbm_model = get_lightgbm_model(seed, **lgbm_params)
        lgbm_model.fit(X_train, y_train)
        oof_preds[val_idx, 0] = lgbm_model.predict_proba(X_val)[:, 1]

        knn_model = get_knn_model(**knn_params)
        knn_model.fit(X_train, y_train)
        oof_preds[val_idx, 1] = knn_model.predict_proba(X_val)[:, 1]

        X_dnn_train, X_dnn_val = X_dnn.iloc[train_idx], X_dnn.iloc[val_idx]
        oof_preds[val_idx, 2] = get_dnn_probs(X_dnn_train, y_train, X_dnn_val, seed, dnn_model_params, dnn_train_params)

    return oof_preds


def evaluate_stacking(oof_preds, y, seed, n_splits, meta_model_type='ridge'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in skf.split(oof_preds, y):
        X_train, X_val = oof_preds[train_idx], oof_preds[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if meta_model_type == 'ridge':
            meta_model = Ridge()
        elif meta_model_type == 'linear':
            meta_model = LinearRegression()

        meta_model.fit(X_train, y_train)
        predictions = (meta_model.predict(X_val) > 0.5).astype(int)
        accuracy = (predictions == y_val.values).mean()
        scores.append(accuracy)

    return scores


def predict_ensemble_averaging(X_train, y_train, X_test, X_dnn_train, X_dnn_test, seed, lgbm_params, knn_params, dnn_model_params, dnn_train_params, weights):
    lgbm_model = get_lightgbm_model(seed, **lgbm_params)
    lgbm_model.fit(X_train, y_train)
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

    knn_model = get_knn_model(**knn_params)
    knn_model.fit(X_train, y_train)
    knn_probs = knn_model.predict_proba(X_test)[:, 1]

    dnn_probs = get_dnn_probs(X_dnn_train, y_train, X_dnn_test, seed, dnn_model_params, dnn_train_params)

    w_lgbm, w_knn, w_dnn = weights
    avg_probs = (lgbm_probs * w_lgbm + knn_probs * w_knn + dnn_probs * w_dnn) / sum(weights)
    predictions = (avg_probs > 0.5).astype(int)
    return predictions


def predict_stacking(X_train, y_train, X_test, X_dnn_train, X_dnn_test, oof_preds, seed, lgbm_params, knn_params, dnn_model_params, dnn_train_params, meta_model_type='linear'):
    lgbm_model = get_lightgbm_model(seed, **lgbm_params)
    lgbm_model.fit(X_train, y_train)
    lgbm_probs_test = lgbm_model.predict_proba(X_test)[:, 1]

    knn_model = get_knn_model(**knn_params)
    knn_model.fit(X_train, y_train)
    knn_probs_test = knn_model.predict_proba(X_test)[:, 1]

    dnn_probs_test = get_dnn_probs(X_dnn_train, y_train, X_dnn_test, seed, dnn_model_params, dnn_train_params)

    test_preds = np.column_stack([lgbm_probs_test, knn_probs_test, dnn_probs_test])

    if meta_model_type == 'ridge':
        meta_model = Ridge()
    elif meta_model_type == 'linear':
        meta_model = LinearRegression()

    meta_model.fit(oof_preds, y_train)
    final_predictions = (meta_model.predict(test_preds) > 0.5).astype(int)
    return final_predictions