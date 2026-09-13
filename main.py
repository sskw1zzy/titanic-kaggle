import re
import itertools
import pandas as pd
import torch
from src.data import load_data
from config import SEED, N_SPLITS, AGE_BINS, FARE_QCUT_Q, COMMON_TITLES, SCALING_COLS, DROP_COLS, ONE_HOT_COLS, C_VALUES, N_NEIGHBORS_VALUES, MAX_DEPTH_VALUES, MIN_SAMPLES_LEAF_VALUES, N_ESTIMATORS_VALUES, CATBOOST_ITERATIONS_VALUES, CATBOOST_DEPTH_VALUES, CATBOOST_LR_VALUES, XGB_N_ESTIMATORS_VALUES, XGB_MAX_DEPTH_VALUES, XGB_LR_VALUES, LGBM_N_ESTIMATORS_VALUES, LGBM_MAX_DEPTH_VALUES, LGBM_LR_VALUES, DNN_HIDDEN_SIZES_OPTIONS, DNN_DROPOUT_VALUES, DNN_BATCHNORM_OPTIONS, DNN_ACTIVATIONS, DNN_LR_VALUES, DNN_BATCH_SIZE, DNN_NUM_EPOCHS, DNN_OPTIMIZER_VALUES, DNN_SCHEDULER_VALUES, DNN_BATCH_SIZE_VALUES, DNN_NUM_EPOCHS_VALUES, DNN_HIDDEN_SIZES_RETEST, DNN_LR_VALUES_RETEST, BEST_LGBM_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, BEST_KNN_PARAMS
from src.features import add_has_cabin, add_bins, one_hot_encoding, drop_unused, add_title, add_scaling, add_family
from src.validation import evaluate_model, evaluate_catboost
from src.models import get_baseline_model, get_regularized_model, get_knn_model, get_tree_model, get_forest_model, get_lightgbm_model, get_xgboost_model, get_catboost_model
from src.dnn import DNNModel, prepare_dnn_data, train_dnn, evaluate_dnn, evaluate_dnn_kfold
from src.ensemble import evaluate_ensemble_averaging, evaluate_ensemble_averaging_v2, evaluate_ensemble_4models, generate_oof_predictions, evaluate_stacking, predict_ensemble_averaging, predict_stacking
import torch.nn as nn
import torch.optim as optim

train_df, test_df = load_data()

train_df = add_has_cabin(train_df)
train_df = add_title(train_df, COMMON_TITLES)
train_df, fare_bins = add_bins(train_df, AGE_BINS, FARE_QCUT_Q)
train_df = one_hot_encoding(train_df, ONE_HOT_COLS)
train_df = drop_unused(train_df, DROP_COLS)
train_df = add_family(train_df)
train_df, scaler = add_scaling(train_df, SCALING_COLS)

X = train_df.drop(columns=['Survived', 'PassengerId'])
X.columns = [re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in X.columns]
y = train_df['Survived']

# model = get_baseline_model(SEED)
# scores = evaluate_model(model, X, y, SEED, n_splits=N_SPLITS)
# print('CV scores:', scores)
# print('Mean CV accuracy:', scores.mean())

# for c in C_VALUES:
#     l1_model = get_regularized_model(SEED, penalty='l1', C=c)
#     l1_scores = evaluate_model(l1_model, X, y, SEED, n_splits=N_SPLITS)
#     print(f'L1, C={c}: mean CV = {l1_scores.mean():.4f}')

# l2_model = get_regularized_model(SEED, penalty='l2', C=10.0)
# l2_scores = evaluate_model(l2_model, X, y, SEED, n_splits=N_SPLITS)
# print(f'L2, C=10.0: mean CV = {l2_scores.mean():.4f}')

# for k in N_NEIGHBORS_VALUES:
#     knn_model = get_knn_model(n_neighbors=k)
#     knn_scores = evaluate_model(knn_model, X, y, SEED, n_splits=N_SPLITS)
#     print(f'KNN, k={k}: mean CV = {knn_scores.mean():.4f}')

# for depth in MAX_DEPTH_VALUES:
#     for leaf in MIN_SAMPLES_LEAF_VALUES:
#         tree_model = get_tree_model(SEED, max_depth=depth, min_samples_leaf=leaf)
#         tree_scores = evaluate_model(tree_model, X, y, SEED, n_splits=N_SPLITS)
#         print(f'Tree, depth={depth}, leaf={leaf}: mean CV = {tree_scores.mean():.4f}')

# forest_results = []
# for n_est in N_ESTIMATORS_VALUES:
#     for depth in MAX_DEPTH_VALUES:
#         for leaf in MIN_SAMPLES_LEAF_VALUES:
#             forest_model = get_forest_model(SEED, n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf)
#             forest_scores = evaluate_model(forest_model, X, y, SEED, n_splits=N_SPLITS)
#             forest_results.append({
#                 'n_estimators': n_est,
#                 'max_depth': depth,
#                 'min_samples_leaf': leaf, 
#                 'mean_cv': forest_scores.mean()
#             })

# forest_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in forest_results[:5]:
#     print(r)

# train_df_cb = load_data()[0]
# train_df_cb = add_has_cabin(train_df_cb)
# train_df_cb = add_title(train_df_cb, COMMON_TITLES)
# train_df_cb, _ = add_bins(train_df_cb, AGE_BINS, FARE_QCUT_Q)
# train_df_cb = drop_unused(train_df_cb, DROP_COLS)
# train_df_cb = add_family(train_df_cb)
# train_df_cb, _ = add_scaling(train_df_cb, SCALING_COLS)

# cat_features = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']
# train_df_cb[cat_features] = train_df_cb[cat_features].astype(str)

# X_cb = train_df_cb.drop(columns=['Survived', 'PassengerId'])
# y_cb = train_df_cb['Survived']

train_df_ext = load_data()[0]
train_df_ext = add_has_cabin(train_df_ext)
train_df_ext = add_title(train_df_ext, COMMON_TITLES)
train_df_ext, _ = add_bins(train_df_ext, AGE_BINS, FARE_QCUT_Q)
train_df_ext = one_hot_encoding(train_df_ext, ONE_HOT_COLS)
train_df_ext = drop_unused(train_df_ext, DROP_COLS)
train_df_ext = add_family(train_df_ext)
train_df_ext['Pclass_x_Family'] = train_df_ext['Pclass'] * train_df_ext['Family']
train_df_ext, _ = add_scaling(train_df_ext, SCALING_COLS)

X_ext = train_df_ext.drop(columns=['Survived', 'PassengerId'])
X_ext.columns = [re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in X_ext.columns]

# catboost_results = []
# for iters in CATBOOST_ITERATIONS_VALUES:
#     for depth in CATBOOST_DEPTH_VALUES:
#         for lr in CATBOOST_LR_VALUES:
#             cb_scores = evaluate_catboost(X_cb, y_cb, cat_features, SEED, n_splits=N_SPLITS, 
#                                           iterations=iters, depth=depth, learning_rate=lr)
#             catboost_results.append({
#                 'iterations': iters,
#                 'depth': depth, 
#                 'learning_rate': lr,
#                 'mean_cv': sum(cb_scores) / len(cb_scores)
#             })

# catboost_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in catboost_results[:5]:
#     print(r)

# xgb_results = []
# for n_est in XGB_N_ESTIMATORS_VALUES:
#     for depth in XGB_MAX_DEPTH_VALUES:
#         for lr in XGB_LR_VALUES:
#             xgb_model = get_xgboost_model(SEED, n_estimators=n_est, max_depth=depth, learning_rate=lr)
#             xgb_scores = evaluate_model(xgb_model, X, y, SEED, n_splits=N_SPLITS)
#             xgb_results.append({'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr, 'mean_cv': xgb_scores.mean()})

# xgb_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in xgb_results[:5]:
#     print(r)

# lgbm_results = []
# for n_est in LGBM_N_ESTIMATORS_VALUES:
#     for depth in LGBM_MAX_DEPTH_VALUES:
#         for lr in LGBM_LR_VALUES:
#             lgbm_model = get_lightgbm_model(SEED, n_estimators=n_est, max_depth=depth, learning_rate=lr)
#             lgbm_scores = evaluate_model(lgbm_model, X, y, SEED, n_splits=N_SPLITS)
#             lgbm_results.append({'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr, 'mean_cv': lgbm_scores.mean()})

# lgbm_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in lgbm_results[:5]:
#     print(r)

# cb_onehot_model = get_catboost_model(SEED, iterations=100, depth=8, learning_rate=0.1)
# cb_onehot_scores = evaluate_model(cb_onehot_model, X, y, SEED, n_splits=N_SPLITS)
# print('CatBoost on One-Hot X: mean CV =', cb_onehot_scores.mean())

# dnn_grid = itertools.product(DNN_HIDDEN_SIZES_OPTIONS, DNN_DROPOUT_VALUES, DNN_BATCHNORM_OPTIONS, DNN_ACTIVATIONS, DNN_LR_VALUES)

# dnn_results = []
# for hidden_sizes, dropout_p, use_bn, activation, lr in dnn_grid:
#     model_params = {'hidden_sizes': hidden_sizes, 'dropout_p': dropout_p, 'use_batchnorm': use_bn, 'activation': activation}
#     train_params = {'batch_size': DNN_BATCH_SIZE, 'lr': lr, 'num_epochs': DNN_NUM_EPOCHS}
    
#     scores = evaluate_dnn_kfold(X, y, SEED, N_SPLITS, model_params=model_params, train_params=train_params)
    
#     dnn_results.append({**model_params, 'lr': lr, 'mean_cv': sum(scores) / len(scores)})
    
# dnn_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in dnn_results[:10]:
#     print(r)

# training_grid = itertools.product(DNN_OPTIMIZER_VALUES, DNN_SCHEDULER_VALUES, DNN_BATCH_SIZE_VALUES,
#                                   DNN_NUM_EPOCHS_VALUES, DNN_LR_VALUES_RETEST, DNN_HIDDEN_SIZES_RETEST)

# training_results = []
# for opt_name, sched_name, batch_size, num_epochs, lr, hidden_sizes in training_grid:
#     model_params = {'hidden_sizes': hidden_sizes, 'dropout_p': 0.5, 'use_batchnorm': False, 'activation': 'leaky_relu'}
#     train_params = {'batch_size': batch_size, 'lr': 0.01, 'num_epochs': num_epochs}
    
#     scores = evaluate_dnn_kfold(X, y, SEED, N_SPLITS, model_params=model_params, 
#                                 train_params=train_params, optimizer_name=opt_name, scheduler_name=sched_name)
    
#     training_results.append({'optimizer': opt_name, 'scheduler': sched_name, 'batch_size': batch_size,
#                              'num_epochs': num_epochs, 'lr': lr, 'hidden_sizes': hidden_sizes,
#                              'mean_cv': sum(scores) / len(scores)})

# training_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in training_results[:10]:
#     print(r)

# best_lgbm = get_lightgbm_model(SEED, n_estimators=100, max_depth=3, learning_rate=0.05)
# best_lgbm.fit(X, y)

# importances = pd.Series(best_lgbm.feature_importances_, index=X.columns)
# importances = importances.sort_values(ascending=False)
# print(importances.head(15))

# weight_options = [(1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 1, 4), (5, 1, 5), (3, 0.5, 3), (4, 1, 3), (3, 1, 4), (1, 0, 1)]

# for weights in weight_options:
#     scores = evaluate_ensemble_averaging(X, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, weights=weights)
#     print(f'Weights {weights}: mean CV = {sum(scores)/len(scores):.4f}')

# knn_weight_options = [(1.5, 1, 0), (1, 1.5, 0), (2, 1, 0), (1, 2, 0.5), (1, 2, 0), (2.5, 2, 0.5), (1, 2.5, 0)]

# for weights in knn_weight_options:
#     scores = evaluate_ensemble_averaging_v2(X, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, weights=weights)
#     print(f'KNN ensemble weights {weights}: mean CV = {sum(scores)/len(scores):.4f}')

# big_weight_options = [
#     (1, 1.1, 0), (1, 1.2, 0), (1, 1.3, 0), (1, 1.4, 0), (1, 1.6, 0),
#     (1, 1.7, 0), (1, 1.8, 0), (1, 1.9, 0), (1, 2, 0),
#     (0.5, 1.5, 0), (0.8, 1.5, 0), (0.9, 1.5, 0), (1.1, 1.5, 0), (1.2, 1.5, 0),
#     (1, 1.5, 0.1), (1, 1.5, 0.2), (1, 1.4, 0.1), (1, 1.6, 0.1),
#     (1, 3, 0), (1, 1.5, 1),
# ]

# big_results = []
# for weights in big_weight_options:
#     scores = evaluate_ensemble_averaging_v2(X, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, weights=weights)
#     big_results.append({'weights': weights, 'mean_cv': sum(scores) / len(scores)})

# big_results.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in big_results:
#     print(r)

# weights_4models_options = [
#     (1, 1.5, 1, 0.1), (1, 1.5, 0.5, 0.1), (1, 1.5, 1.5, 0.1), (1, 1.5, 2, 0.1),
#     (1, 1.5, 1, 0), (1, 1.5, 0.5, 0), (1, 1.5, 1.5, 0), (1, 1.5, 2, 0),
#     (1, 1.5, 1, 0.2), (1, 1.5, 0.3, 0.1), (1, 1.5, 0.7, 0.1),
#     (1.5, 1.5, 1, 0.1), (0.7, 1.5, 1, 0.1), (1, 2, 1, 0.1), (1, 1, 1, 0.1),
#     (1, 1.5, 1, 0.1), (2, 1.5, 1, 0.1), (1, 1.5, 3, 0.1), (1, 1.5, 0.1, 0.1),
#     (1, 1.5, 1, 0.3), (1, 1.5, 1, 0.05),
# ]

# results_4models = []
# for weights in weights_4models_options:
#     scores = evaluate_ensemble_4models(X, X_ext, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, weights=weights)
#     results_4models.append({'weights': weights, 'mean_cv': sum(scores) / len(scores)})

# results_4models.sort(key=lambda r: r['mean_cv'], reverse=True)
# for r in results_4models[:15]:
#     print(r)

# oof_preds = generate_oof_predictions(X, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS)

# for meta_type in ['linear', 'ridge']:
#     stack_scores = evaluate_stacking(oof_preds, y, SEED, N_SPLITS, meta_model_type=meta_type)
#     print(f'Stacking ({meta_type}): mean CV = {sum(stack_scores)/len(stack_scores):.4f}')

test_df_processed = add_has_cabin(test_df)
test_df_processed = add_title(test_df_processed, COMMON_TITLES)
test_df_processed, _ = add_bins(test_df_processed, AGE_BINS, FARE_QCUT_Q, fare_bins=fare_bins)
test_df_processed = one_hot_encoding(test_df_processed, ONE_HOT_COLS)
test_df_processed = drop_unused(test_df_processed, DROP_COLS)
test_df_processed = add_family(test_df_processed)
test_df_processed, _ = add_scaling(test_df_processed, SCALING_COLS, scaler=scaler)

test_ids = test_df_processed['PassengerId']
X_test = test_df_processed.drop(columns=['PassengerId'])
X_test.columns = [re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in X_test.columns]
print(X_test.shape, X.shape)

# final_predictions = predict_ensemble_averaging(
#     X, y, X_test, X, X_test, SEED,
#     BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS,
#     weights=(1, 1.5, 0.1)
# )

# submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': final_predictions})
# submission.to_csv('submission_ensemble.csv', index=False)

# oof_preds = generate_oof_predictions(X, y, X, SEED, N_SPLITS, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS)

# stacking_predictions = predict_stacking(X, y, X_test, X, X_test, oof_preds, SEED, BEST_LGBM_PARAMS, BEST_KNN_PARAMS, BEST_DNN_MODEL_PARAMS, BEST_DNN_TRAIN_PARAMS, meta_model_type='linear')

# submission_stacking = pd.DataFrame({'PassengerId': test_ids, 'Survived': stacking_predictions})
# submission_stacking.to_csv('submission_stacking.csv', index=False)

# sub1 = pd.read_csv('submission_ensemble.csv')
# sub2 = pd.read_csv('submission_stacking.csv')
# print((sub1['Survived'] == sub2['Survived']).mean())

# lgbm_final = get_lightgbm_model(SEED, **BEST_LGBM_PARAMS)
# lgbm_final.fit(X, y)
# print('LightGBM train accuracy:', lgbm_final.score(X, y))

# knn_final = get_knn_model(**BEST_KNN_PARAMS)
# knn_final.fit(X, y)
# print('KNN train accuracy:', knn_final.score(X, y))

# torch.manual_seed(SEED)
# train_loader = prepare_dnn_data(X, y, batch_size=BEST_DNN_TRAIN_PARAMS['batch_size'])
# dnn_final = DNNModel(input_size=X.shape[1], **BEST_DNN_MODEL_PARAMS)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(dnn_final.parameters(), lr=BEST_DNN_TRAIN_PARAMS['lr'])
# dnn_final = train_dnn(dnn_final, train_loader, loss_fn, optimizer, BEST_DNN_TRAIN_PARAMS['num_epochs'])
# dnn_train_accuracy = evaluate_dnn(dnn_final, train_loader)
# print('DNN train accuracy:', dnn_train_accuracy)

# comparison = pd.DataFrame({
#     'train_mean': X.mean(),
#     'test_mean': X_test.mean(),
#     'train_std': X.std(),
#     'test_std': X_test.std()
# })
# comparison['mean_diff'] = (comparison['train_mean'] - comparison['test_mean']).abs()
# comparison = comparison.sort_values('mean_diff', ascending=False)
# print(comparison)

# print(X.columns.tolist() == X_test.columns.tolist())

# sample_train_row = train_df.iloc[[0]]
# print(sample_train_row[['Pclass', 'SibSp', 'Parch', 'Family']])

# sample_test_row = test_df_processed.iloc[[0]]
# print(sample_test_row.columns.tolist())
# print(X.dtypes.equals(X_test.dtypes))

# logreg_final = get_baseline_model(SEED)
# logreg_final.fit(X, y)
# logreg_test_predictions = logreg_final.predict(X_test)

# submission_logreg = pd.DataFrame({'PassengerId': test_ids, 'Survived': logreg_test_predictions})
# submission_logreg.to_csv('submission_logreg_simple.csv', index=False)

# lgbm_solo = get_lightgbm_model(SEED, **BEST_LGBM_PARAMS)
# lgbm_solo.fit(X, y)
# lgbm_solo_predictions = lgbm_solo.predict(X_test)

# submission_lgbm_solo = pd.DataFrame({'PassengerId': test_ids, 'Survived': lgbm_solo_predictions})
# submission_lgbm_solo.to_csv('submission_lgbm_solo.csv', index=False)

# embarked_cols = [col for col in X.columns if col.startswith('Embarked')]
# X_no_embarked = X.drop(columns=embarked_cols)
# X_test_no_embarked = X_test.drop(columns=embarked_cols)

# lgbm_no_emb_cv = evaluate_model(get_lightgbm_model(SEED, **BEST_LGBM_PARAMS), X_no_embarked, y, SEED, n_splits=N_SPLITS)
# print('LightGBM without Embarked, CV:', lgbm_no_emb_cv.mean())

# lgbm_no_emb = get_lightgbm_model(SEED, **BEST_LGBM_PARAMS)
# lgbm_no_emb.fit(X_no_embarked, y)
# lgbm_no_emb_predictions = lgbm_no_emb.predict(X_test_no_embarked)

# submission_lgbm_no_emb = pd.DataFrame({'PassengerId': test_ids, 'Survived': lgbm_no_emb_predictions})
# submission_lgbm_no_emb.to_csv('submission_lgbm_no_embarked.csv', index=False)

# knn_solo = get_knn_model(**BEST_KNN_PARAMS)
# knn_solo.fit(X, y)
# knn_predictions = knn_solo.predict(X_test)
# pd.DataFrame({'PassengerId': test_ids, 'Survived': knn_predictions}).to_csv('submission_knn_solo.csv', index=False)

# forest_solo = get_forest_model(SEED, n_estimators=50, max_depth=5, min_samples_leaf=1)
# forest_solo.fit(X, y)
# forest_predictions = forest_solo.predict(X_test)
# pd.DataFrame({'PassengerId': test_ids, 'Survived': forest_predictions}).to_csv('submission_forest_solo.csv', index=False)

# torch.manual_seed(SEED)
# dnn_train_loader = prepare_dnn_data(X, y, batch_size=BEST_DNN_TRAIN_PARAMS['batch_size'])
# dnn_solo = DNNModel(input_size=X.shape[1], **BEST_DNN_MODEL_PARAMS)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(dnn_solo.parameters(), lr=BEST_DNN_TRAIN_PARAMS['lr'])
# dnn_solo = train_dnn(dnn_solo, dnn_train_loader, loss_fn, optimizer, BEST_DNN_TRAIN_PARAMS['num_epochs'])
# dnn_solo.eval()
# with torch.no_grad():
#     X_test_tensor = torch.tensor(X_test.astype('float32').values, dtype=torch.float32)
#     dnn_predictions = (torch.sigmoid(dnn_solo(X_test_tensor)) > 0.5).int().squeeze().numpy()
# pd.DataFrame({'PassengerId': test_ids, 'Survived': dnn_predictions}).to_csv('submission_dnn_solo.csv', index=False)

xgb_solo = get_xgboost_model(SEED, n_estimators=300, max_depth=3, learning_rate=0.05)
xgb_solo.fit(X, y)
xgb_predictions = xgb_solo.predict(X_test)
pd.DataFrame({'PassengerId': test_ids, 'Survived': xgb_predictions}).to_csv('submission_xgb_solo.csv', index=False)

forest_soft = get_forest_model(SEED, n_estimators=500, max_depth=5, min_samples_leaf=5)
forest_soft.fit(X, y)
forest_soft_predictions = forest_soft.predict(X_test)
pd.DataFrame({'PassengerId': test_ids, 'Survived': forest_soft_predictions}).to_csv('submission_forest_soft.csv', index=False)