# Kumo Tabular vs tuned baselines

**Can a tabular foundation model that never trains beat tuned XGBoost?**

[Kumo Tabular](https://kumo.ai) predicts from labelled rows passed as context, with no training. This notebook compares it with XGBoost and logistic regression, both tuned with cross-validation, on the UCI Bank Marketing dataset (will a customer subscribe to a term deposit?), as the number of labelled rows grows.

## Setup

- **Data:** Bank Marketing, 41,188 rows, 11.3% positive. The `duration` column (call length) is dropped, because it is only known after the outcome.
- **Test set:** 4,000 fixed rows.
- **Labelled rows:** n = 20 to 5,000 rows, sampled from the rest and repeated over several seeds.
- **Baselines:** tuned with stratified cross-validation on the labelled rows only, so no model sees the test set.
- **Metric:** ROC-AUC on the test set, plus the share of all "yes" customers found in the top 10% of scores.

## Results (mean ROC-AUC)

| Labelled rows | Kumo Tabular | XGBoost (tuned) | XGBoost (default) | LogReg (tuned) |
|---|---|---|---|---|
| 20 | **0.668** | 0.575 | 0.500 | 0.666 |
| 50 | **0.731** | 0.723 | 0.686 | 0.719 |
| 100 | 0.735 | 0.711 | 0.708 | **0.740** |
| 500 | **0.770** | 0.750 | 0.717 | 0.756 |
| 1,000 | **0.776** | 0.769 | 0.720 | 0.760 |
| 5,000 | **0.786** | 0.778 | 0.738 | 0.772 |

Kumo Tabular matches or beats the best tuned baseline at every size except 100 rows, without any training or tuning. The margin is small (under 0.015 AUC). The clearest win is over XGBoost with very little data, where tree models cannot learn much from 20 rows.

## Run it

Open [`kumo_vs_xgboost_bank_marketing.ipynb`](kumo_vs_xgboost_bank_marketing.ipynb) in Google Colab with a GPU. It downloads the data and the model itself, and takes about 10–20 minutes because of the baseline tuning.

## Stack

Python · Kumo Tabular · XGBoost · scikit-learn · pandas · matplotlib
