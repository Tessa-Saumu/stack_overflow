# Stack Overflow Developer Survey 2025
### Does AI tool adoption independently predict developer salary — or is experience the real driver?

---

## The Question

The Stack Overflow Developer Survey 2025 shows that developers who use AI tools earn more than those who don't. This project asks whether that premium holds up after controlling for the variables most likely to explain it: work experience, region, developer type, industry, and tech stack.

The short answer: it doesn't. The longer answer is more interesting.

---

## Project Structure

```
STACK_OVERFLOW_SURVEY/
│
├── notebooks/
│   ├── data/                          # Parquet artefacts from NB1 (train/test splits)
│   ├── models/
│   │   ├── classification/            # Saved classifier pipeline
│   │   └── regression/                # Saved regression pipeline
│   │
│   ├── eda_notebook.ipynb             # Exploratory analysis & confound identification
│   ├── notebook_1_data_pipeline_setup.ipynb    # Cleaning, feature engineering, train/test split
│   ├── notebook_2_salary_regression_pipelines.ipynb  # Salary regression (Ridge / RF / XGBoost)
│   └── notebook_3_ai_adoption_classifier_pipelines.ipynb  # AI adoption classifier (LR / RF / XGBoost)
│
├── visuals/
│   ├── EDA_Results/                   # Confound analysis plots, salary distributions
│   └── Model_Results/                 # CV comparison, SHAP plots, confusion matrices
│
├── Data Cleaning and EDA Summary.md   # Written summary of EDA findings
└── Model Summary.md                   # Written summary of modelling findings (this file)
```

---

## Notebooks — Read in Order

| # | Notebook | What it does |
|---|---|---|
| EDA | `eda_notebook.ipynb` | Explores salary distributions, AI adoption rates by experience tier, and identifies the confound hypothesis |
| 1 | `notebook_1_data_pipeline_setup.ipynb` | Cleans data, engineers features, produces locked train/test parquet splits |
| 2 | `notebook_2_salary_regression_pipelines.ipynb` | Runs three regression pipelines, selects winner, evaluates on test set, runs SHAP |
| 3 | `notebook_3_ai_adoption_classifier_pipelines.ipynb` | Runs three classification pipelines, selects winner, evaluates on test set, runs SHAP and LR coefficients |

Notebooks 2 and 3 load directly from the parquet artefacts produced by Notebook 1 — no re-splitting.

---

## Approach

**Two models. One research question.**

A salary regression model (Notebook 2) quantifies the marginal contribution of `UsesAI` after all other features are controlled. A binary classifier (Notebook 3) models AI adoption itself, to determine whether experience predicts adoption — and in which direction.

Both models use `sklearn.Pipeline` with preprocessing fitted inside cross-validation folds to prevent leakage. Model selection is programmatic: primary metric first, fold consistency as tiebreaker.

| Task | Models Compared | Winner | Primary Metric |
|---|---|---|---|
| Salary regression | Ridge · Random Forest · XGBoost | Random Forest | CV MAE |
| AI adoption classification | Logistic Regression · Random Forest · XGBoost | Logistic Regression | CV ROC-AUC |

---

## Results

### Regression (Notebook 2)

| Metric | Value |
|---|---|
| Test MAE (log scale) | 0.5233 |
| Test R² | 0.4553 |
| Approx. USD error at median salary | ±$54,247 |

The model explains ~46% of salary variance across 177 countries. The USD error figure makes clear this is a ballpark estimator, not a precision tool.

**SHAP — UsesAI:**

```
Rank:                   46 / 103
% of total attribution: 0.28%
Direction (AI = 1):    −0.00102   ← negative after controls
```

The apparent salary premium for AI users does not survive the model. Once tech stack is controlled, `UsesAI` contributes less than 0.3% of total SHAP attribution — and in a marginally negative direction.

### Classification (Notebook 3)

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.6984 |
| Test Accuracy | 0.6325 |
| Naive baseline accuracy | 0.8163 |

Accuracy falls below the naive baseline by design — `class_weight='balanced'` trades majority-class accuracy for minority-class recall. ROC-AUC is the informative metric here.

**WorkExp coefficient (Logistic Regression):**

```
Coefficient:  −0.1518
Odds ratio:    0.859
Interpretation: each additional year of experience reduces AI adoption odds by ~14%
```

Senior developers are less likely to use AI tools — confirmed by both the logistic regression coefficient and SHAP direction. The confound runs opposite to the naive assumption.

---

## Key Finding

EDA within US experience tiers showed a $9,000–$18,500 salary premium for AI users. The regression model — trained globally with tech stack controls — reduces `UsesAI` to 46th place with a negative SHAP direction.

The premium is absorbed by enterprise tool usage. Developers who use AI tools disproportionately also use Kubernetes, Snowflake, and AWS — which independently predict higher salaries. AI adoption is a proxy for a broader high-salary developer profile, not a salary driver in its own right.

---

## Limitations

- R² of 0.4553 reflects the difficulty of global salary prediction across purchasing-power-different markets. Not suitable for precision estimates.
- EDA confound analysis used US-only data; the regression model is trained globally.
- No hyperparameter tuning was applied to any model (Ridge `alpha=1.0` fixed; RF and XGBoost use library defaults).
- `AISelect` is binarised from a multi-select column — adoption intensity is not captured.
- SHAP values are approximated from a 2,000-row test sample.

---

## Stack

- Python 3.11
- scikit-learn · XGBoost · SHAP
- pandas · numpy · matplotlib · seaborn
- Data: [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/)