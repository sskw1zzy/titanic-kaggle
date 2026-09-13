# Results

## Model families (best CV per family)

| Family | Best config | CV |
|---|---|---|
| Logistic regression | L1/L2, C≥10 | 0.8283 |
| KNN | k=9 | 0.8316 |
| Decision Tree | depth=5 | 0.8272 |
| Random Forest | n=50, depth=5 | 0.8350 |
| CatBoost | native categorical features | 0.8384 |
| XGBoost | n=100, depth=3, lr=0.05 | 0.8406 |
| LightGBM | n=100, depth=3, lr=0.05 | 0.8406 |
| DNN | best of 216-config search | 0.8417 |
| Ensemble (weighted averaging) | LGBM+KNN+DNN | 0.8440 |
| Stacking | Ridge meta-model | 0.8395 |

## Submissions, by LB score

| Model | LB |
|---|---|
| **Random Forest (n=50, depth=5)** | **0.7823 — final model** |
| Random Forest, softer (n=500, leaf=5) | 0.7799 |
| LightGBM, Embarked removed | 0.7751 |
| Baseline logreg | 0.7727 |
| DNN | 0.7679 |
| LightGBM | 0.7679 |
| Random Forest (n=50, depth=5) same config, solo | 0.7679 |
| Ensemble (weighted averaging) | 0.7608 |
| Stacking | 0.7608 |
| XGBoost | 0.7560 |
| KNN | 0.7464 |

## Feature engineering

- `Title` (extracted from `Name`) — strong signal, second only to `Pclass` in LightGBM feature importance.
- `HasCabin` (binary, `Cabin` was 77% missing) — missingness itself correlates with `Pclass`, kept as a feature.
- `Pclass × Family` interaction — helps linear models (+0.01 to +0.02), neutral/negative on trees, boosting, KNN. Not in the final feature set.

## CV → LB gap

Every model scored 5-8 points lower on the public leaderboard than in local CV.
Diagnosed as three separate causes, not a bug (full breakdown in README): distribution
shift on `Embarked`, a small test set amplifying variance, and mild overfitting to the
CV folds from manual weight/hyperparameter search — the more tuned a model was
(ensemble, DNN), the worse it did on LB relative to CV. Titanic is also a well-known case
where LB scores above ~0.90 usually mean leaked historical data rather than a genuinely
better model, so 0.77-0.80 is a realistic ceiling for an honest solution.
