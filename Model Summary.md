# Stack Overflow Developer Survey 2025
## Model Summary — Salary Regression & AI Adoption Classifier

> **Research question:** Does AI tool adoption independently predict developer salary, or is experience the true driver — with AI usage as a correlated side-effect?

---

## 1. Objective

This project tests a specific hypothesis: the observed salary premium for AI-tool-using developers (documented in raw survey data) may be an artefact of confounding. Senior developers earn more *and* are more likely to use certain tools — including AI tools. If experience and tech stack choices explain the salary gap, then AI adoption itself carries no independent signal.

Two complementary models were built to answer this:

- **Regression (Notebook 2):** Predict log salary, then use SHAP to quantify the marginal contribution of `UsesAI` after controlling for all other features.
- **Classifier (Notebook 3):** Predict AI adoption from structural developer features, then use SHAP and logistic regression coefficients to determine whether experience suppresses or drives adoption.

The findings from both models converge on the same answer — detailed in [Key Insights](#5-key-insights).

---

## 2. Modelling Approach

### Data

- **Source:** Stack Overflow Developer Survey 2025
- **Scope:** Global respondents with valid salary and AI adoption responses
- **Train / Test split:** 14,411 training rows · 3,603 test rows (held out from Notebook 1)
- **Target — Regression:** `log_salary` (log-transformed to correct right skew)
- **Target — Classifier:** `UsesAI` (binary: 1 = uses AI tools, 0 = does not)

### Feature Engineering

Features were selected based on EDA thresholds: a ±15% salary premium test (regression features) and a minimum 500-user frequency filter. Core controls applied in both models: `WorkExp`, `Region` (10 geographic clusters), `DevType`, `Industry`, and `ExperienceTier`. Tech stack features (44 columns) were included in the regression to absorb tool-choice variance. `YearsCode` was dropped due to multicollinearity with `WorkExp` (Pearson r > 0.85).

### Pipeline Architecture

Both notebooks use `sklearn.Pipeline` with `ColumnTransformer` preprocessing. `WorkExp` is scaled via `StandardScaler` inside the pipeline — fitted on training folds only, preventing leakage. Leakage assertions run at notebook load:

```python
assert 'log_salary' not in X_train.columns   # regression
assert 'UsesAI' not in X_clf_train.columns   # classifier
```

### Cross-Validation Strategy

| Task | CV Strategy | Primary Metric | Rationale |
|---|---|---|---|
| Regression | `KFold(n=5)` | CV MAE | Symmetric error metric; no class imbalance |
| Classification | `StratifiedKFold(n=5)` | CV ROC-AUC | 81.6% positive class; accuracy is uninformative |

Model selection uses a tolerance-based tiebreaker: if the MAE/AUC gap between candidates falls within ±0.005, the more consistent model (lower ±std across folds) is preferred.

---

## 3. Regression Results

### Model Comparison (5-Fold CV)

| Model | CV MAE | ±std | CV R² | ±std | Train MAE | Overfit Gap |
|---|---|---|---|---|---|---|
| Ridge | 0.5589 | 0.0162 | 0.4197 | 0.0239 | 0.5533 | +0.0056 |
| **Random Forest** | **0.5263** | **0.0138** | **0.4512** | **0.0216** | 0.4706 | +0.0557 |
| XGBoost | 0.5279 | 0.0170 | 0.4511 | 0.0289 | 0.4179 | +0.1100 |

*Overfit gap = CV MAE − Train MAE (positive = overfitting).*

**Winner: Random Forest** — lowest CV MAE (0.5263), lowest fold variance (±0.0138). XGBoost's marginal MAE improvement (0.0016) fell within the tiebreaker tolerance; its overfit gap of 0.1100 vs Random Forest's 0.0557 confirmed the selection.

### Final Test Set Results

| Metric | Value |
|---|---|
| Test MAE (log scale) | 0.5233 |
| Test RMSE (log scale) | 0.8327 |
| Test R² | 0.4553 |
| Approx. USD error at median ($78,890) | ±$54,247 |

**Interpretation:** The model explains ~46% of global salary variance. For a survey-based model trained across 177 countries with structural purchasing-power differences, this is a defensible result — but it should not be presented as a precision tool. The ±$54,247 USD approximation makes this appropriate for ballpark range estimates, not point predictions.

### SHAP — UsesAI Marginal Effect

> **This is the central finding of the project.**

```
UsesAI SHAP rank:        46 / 103
% of total attribution:  0.28%
Direction (AI = 1):     −0.00102   ← negative
Direction (AI = 0):     +0.00447
```

After controlling for `WorkExp`, `Region`, `DevType`, `Industry`, and 44 tech stack features, **AI users receive a marginally negative SHAP contribution to salary prediction.** `UsesAI` ranks 46th out of 103 features — statistically indistinguishable from noise.

This directly contradicts the raw EDA finding of a $9,000–$18,500 salary premium for AI users within experience tiers (US data). The SHAP result reveals that this premium is absorbed by tech stack controls: developers who use AI tools also disproportionately use high-salary enterprise tools (Snowflake, Kubernetes, AWS). Once those tool choices are controlled for, the marginal contribution of `UsesAI` alone collapses to near zero.

---

## 4. Classification Results

### Model Comparison (5-Fold Stratified CV)

| Model | CV ROC-AUC | ±std | CV F1 | ±std | Train AUC | Overfit Gap |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.6837** | **0.0093** | 0.7227 | 0.0056 | 0.7035 | −0.0198 |
| Random Forest | 0.6782 | 0.0112 | **0.8229** | 0.0046 | 0.9066 | −0.2284 |
| XGBoost | 0.6707 | 0.0156 | 0.7503 | 0.0071 | 0.8268 | −0.1561 |

*Overfit gap = CV AUC − Train AUC (negative = overfitting).*

> **Note on Random Forest F1 (0.8229):** This figure is misleading. With an 81.6% positive class, a model that mostly predicts the majority class achieves high F1 for the "Uses AI" label without meaningful discriminative power. The RF's Train AUC of 0.9066 vs CV AUC of 0.6782 — a gap of 0.23 — confirms severe overfitting on untuned hyperparameters. Model selection correctly ignores CV F1 in favour of CV ROC-AUC.

**Winner: Logistic Regression** — highest CV ROC-AUC (0.6837), lowest fold variance (±0.0093), negligible overfit gap.

### Final Test Set Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.6984 |
| F1 (macro avg) | 0.57 |
| Accuracy | 0.6325 |
| Naive baseline accuracy | 0.8163 (always predict "Uses AI") |

**Accuracy below the naive baseline is expected.** `class_weight='balanced'` forces the model to actively classify both groups, sacrificing majority-class accuracy in exchange for minority-class (non-AI-adopter) recall. The relevant metric is ROC-AUC = 0.70 — which represents genuine discriminative power above random chance (0.50).

### Classification Report

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No AI (0) | 0.29 | 0.67 | 0.40 |
| Uses AI (1) | 0.89 | 0.62 | 0.73 |
| Macro avg | 0.59 | 0.65 | 0.57 |

The model correctly identifies 67% of actual non-adopters (recall = 0.67), but its minority-class precision is low (0.29) — for every 3 developers predicted as non-adopters, 2 are misclassified. This precision/recall tradeoff reflects the difficulty of identifying a structurally underrepresented group from survey features alone.

### SHAP / Coefficient — WorkExp Direction

```
WorkExp SHAP rank:      6 / 102  (4.32% of attribution)
Direction (high exp):  −0.08374   ← suppresses AI adoption
Direction (low exp):   +0.15185   ← promotes AI adoption

WorkExp LR coefficient: −0.1518  →  OR 0.859
Each additional year of experience reduces AI adoption odds by ~14%
```

Two independent methods — SHAP across tree models and the logistic regression coefficient — confirm the same direction: **experience suppresses AI adoption, not the reverse.** Younger, less experienced developers are the primary AI tool adopters. The confound mechanism the EDA flagged (senior devs inflate AI salaries) is structurally absent: senior devs are less likely to use AI, not more.

Notable DevType coefficients from the logistic regression:

| Developer Type | Direction | Odds Ratio |
|---|---|---|
| AI / ML Developer | Strongly positive | ~3.5× more likely to adopt |
| Game Developer | Strongly negative | ~4× less likely to adopt |

---

## 5. Key Insights

### Finding 1: The AI salary premium disappears after controls

The headline finding of the project is a reversal. Raw EDA shows AI users earn $9,000–$18,500 more per year within experience tiers. SHAP analysis on the fully-controlled regression model shows `UsesAI` contributing a marginally *negative* effect to salary prediction (rank 46/103, 0.28% of total attribution).

**The interpretation:** The apparent premium is explained by the tools developers use alongside AI — enterprise cloud platforms, Kubernetes, Snowflake — which independently predict higher salaries and are disproportionately used by AI adopters. AI adoption is a proxy signal for a broader high-salary developer profile; it is not itself a salary driver.

This is the more honest and more interesting conclusion than "AI users earn more." It also illustrates why regression with controls routinely overturns naive EDA findings.

### Finding 2: Experience suppresses AI adoption

Senior developers are *less* likely to use AI tools. The logistic regression quantifies this at approximately −14% odds per additional year of experience. This finding was consistent across SHAP (non-linear, tree-based) and the logistic regression coefficient (linear, interpretable). The two models agree.

This has an important implication: the salary comparison between AI users and non-users is confounded in an unexpected direction. Non-AI-adopters skew senior (higher salary), which partially compresses the raw premium rather than inflating it.

### Finding 3: Logistic Regression outperformed tree models for behavioural prediction

For a behavioural classification task (AI adoption), the simplest model — Logistic Regression — won on the correct metric (ROC-AUC). Random Forest achieved a high CV F1 but exhibited severe overfitting (Train AUC 0.91, CV AUC 0.68) without hyperparameter tuning. This result supports using interpretable baselines before complex models, especially on survey data with high feature sparsity.

---

## 6. Limitations

| Limitation | Impact |
|---|---|
| R² = 0.4553 (46% variance explained) | Model captures structural salary gradients but misses within-cohort variance. Not suitable for precision salary prediction. |
| ±$54,247 USD error at median salary | Appropriate for range estimates only. A Streamlit deployment should display prediction intervals, not point values. |
| SHAP sampled from 2,000 test rows | SHAP values are an approximation; full-dataset computation was computationally prohibitive. |
| No AI precision = 0.29 | The classifier struggles to precisely identify non-adopters — a known tradeoff from `class_weight='balanced'`. |
| EDA confound check is US-only | The $9k–$18.5k within-tier premium figures apply to US respondents only; the regression model was trained globally. |
| Hyperparameters not tuned | Ridge `alpha=1.0` (fixed baseline); RF and XGBoost use sensible defaults without grid search. |
| `AISelect` binary simplification | AI adoption is encoded as binary from a multi-select column. Nuances in adoption type (e.g., casual vs. daily use) are not captured. |

---

## 7. Final Conclusion

This project set out to test whether AI tool adoption independently predicts developer salary. The answer, after modelling, is no — at least not in isolation.

The regression model (Random Forest, R² = 0.4553) confirms that once experience, region, developer type, and tech stack are controlled, `UsesAI` contributes less than 0.3% of total SHAP attribution. The apparent salary premium observed in EDA is absorbed by the correlated features that more directly predict earnings.

The classifier (Logistic Regression, ROC-AUC = 0.70) reveals that AI adoption is primarily a behaviour of less experienced developers. Each additional year of experience reduces adoption odds by approximately 14%. This makes the raw EDA premium even harder to attribute to AI: the non-adopter group skews senior, partially offsetting the observed gap.

The practical takeaway: **tool choice predicts salary, not AI adoption per se.** Developers who use cloud-native enterprise tools, irrespective of AI adoption, earn more. AI tool usage correlates with those developers — but is not the causal lever.

---

*Data: Stack Overflow Developer Survey 2025. Models: sklearn Pipeline + XGBoost + SHAP. Notebooks: NB1 (data pipeline), NB2 (regression), NB3 (classifier).*