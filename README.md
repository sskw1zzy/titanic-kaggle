# Titanic — Survival Prediction

Kaggle Titanic competition. Final model: Random Forest, LB score **0.7823**.

## Setup

```
pip install -r requirements.txt
```

Download `train.csv`/`test.csv` from the [competition page](https://www.kaggle.com/competitions/titanic/data) into `data/`.

## Run

```
python main.py
```

Produces `submission.csv`. Everything is deterministic (`SEED=42` in `config.py`).

## Pipeline

`main.py` loads the data, runs it through the feature engineering steps below, trains a
Random Forest on the full train set, and predicts on test. All transformation functions
live in `src/features.py`; the same `preprocess()` call is used for both train and test,
with `fare_bins`/`scaler` fit on train and reused (not recomputed) on test to avoid leakage.

## Feature engineering

- `Title` extracted from `Name` (Mr/Mrs/Miss/Master, rare titles grouped) — the strongest
  single feature, catches age/status info that `Sex` alone misses.
- `HasCabin` — binary flag instead of the raw `Cabin` column (77% missing). The missingness
  itself correlates with `Pclass`, so it's informative on its own.
- `Family` (SibSp + Parch + 1) and `IsAlone` — survival isn't linear in family size (small
  families do best, solo and large families do worse), so both the raw count and a binary
  flag are kept.
- `Age`/`Fare` bucketed into bins, then one-hot encoded along with `Sex`/`Embarked`/`Title`.
- Tried `Pclass × Family` as an explicit interaction — helped linear models, neutral/negative
  for trees and boosting (they can already find that interaction on their own). Not included
  in the final feature set. Full numbers in `RESULTS.md`.

## Validation

Stratified 5-fold CV throughout. Chose stratified specifically because the target isn't
perfectly balanced (~62/38).

## Models & ensembling

Tried logistic regression (+ L1/L2/ElasticNet), KNN, decision tree, random forest,
CatBoost/XGBoost/LightGBM, and a custom PyTorch DNN (configurable depth/dropout/batchnorm/
activation, ~200 configs tested). Also tried weighted averaging, voting, and stacking on top
of the strongest models. Full comparison in `RESULTS.md`.

**The final model is Random Forest, not the highest-CV model.** The ensemble scored highest
on CV (0.844) but worst on the actual leaderboard (0.7608). Random Forest wasn't the CV
leader (0.835) but generalized best to the real test set (0.7823) — see below for why.

## The CV → LB gap

Every model scored noticeably lower on the public leaderboard than in local CV — the gap
was consistent (5-8 points) and got *worse* the more tuned/complex the model was. Ran a
proper investigation instead of guessing:

- Ruled out actual bugs first: column order, dtypes, and value distributions between `X`
  and `X_test` all check out. `fare_bins`/`scaler` are correctly fit on train and reused on
  test.
- **Distribution shift**: the "large poor family" pattern (a strong, profitable signal for
  the model) is rarer in test than in train (1.2% vs 3.0%), while the more uncertain cases
  (`Embarked_C`, first-class men) are more common in test. The model applies what it learned,
  but the situations it's most confident about show up less often.
- **Selection bias from tuning**: manually searching ensemble weights over the same 5 folds
  inflates CV by about a point — this is exactly why the ensemble (the most heavily tuned
  model) had the worst CV→LB gap, and why a simple, untuned Random Forest generalized better.
- Titanic is also a well-known case where LB scores above ~0.90 usually mean leaked
  historical passenger data, not a genuinely better model. 0.77-0.80 is a realistic ceiling
  for an honest solution here — this project's 0.7823 sits right in that range.

## Results

Full experiment log and numbers: [`RESULTS.md`](./RESULTS.md).