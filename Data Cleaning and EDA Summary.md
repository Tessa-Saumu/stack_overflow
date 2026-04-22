# Stack Overflow Developer Survey 2025
## Data Cleaning & EDA Summary — v2 (post bug-fix)

---

## Section 1 — Data Cleaning

### 1.1 Column Selection

From the 177-column raw survey, 15 columns were retained based on relevance to the research question and model requirements:

| Column | Role |
|---|---|
| `ResponseId` | Row integrity check |
| `MainBranch` | Filter: professional developers only |
| `EdLevel` | Secondary confounder (evaluated, excluded from model — see 1.6) |
| `Employment` | Context variable |
| `WorkExp` | Primary confounder — experience tier basis |
| `YearsCode` | Dropped post-correlation check (see 1.4) |
| `DevType` | Secondary confounder — seniority proxy |
| `Industry` | Secondary confounder |
| `Country` | Structural confounder — geographic salary baseline |
| `LanguageHaveWorkedWith` | Feature (multi-select → binarized) |
| `DatabaseHaveWorkedWith` | Feature (multi-select → binarized) |
| `PlatformHaveWorkedWith` | Feature (multi-select → binarized) |
| `WebframeHaveWorkedWith` | Feature (multi-select → binarized) |
| `AISelect` | Target variable (classifier) / Feature (salary regressor) |
| `ConvertedCompYearly` | Target variable (salary regressor) |

The 162 excluded columns contain question-level granularity (tool satisfaction scores, company size, etc.) that adds noise without salary-predictive signal at this scale.

---

### 1.2 Corruption Detection & Row Removal

A custom cleaning function addressed two types of row corruption common in large CSV exports of survey data with free-text fields:

- **Orphaned continuation rows** — rows with no valid `ResponseId` after numeric coercion, indicating broken multi-line fields from newline characters in free-text responses
- **Precursor rows** — valid rows immediately followed by an orphaned row, indicating the start of a corruption block

Both types were dropped. Critical columns (`WorkExp`, `AISelect`, `ConvertedCompYearly`) were required to be non-null — rows missing any of these cannot contribute to either model. Non-critical categorical columns (`EdLevel`, `Employment`, `DevType`, `Industry`, `Country`) had NaN values filled with `'Unknown'` to preserve the row's remaining data.

**Data retention:** 96.58% of unique `ResponseId` values retained after corruption removal and critical column filtering.

---

### 1.3 Professional Developer Filter

Filtered to respondents identifying as `'I am a developer by profession'` in `MainBranch`. Hobbyists, students, and non-coding professionals have structurally different salary and AI adoption patterns outside the scope of the research question.

---

### 1.4 Feature Engineering

**Experience Tier (`ExperienceTier`)**

`WorkExp` binned into four ordinal tiers using `pd.cut` with `bins=[-1, 2, 5, 10, 100]`. The lower bound of -1 ensures zero-year respondents are captured in the junior tier rather than being excluded. Assignment references `df_pros['WorkExp']` to ensure correct row alignment after the professional developer filter and index reset.

| Tier | Experience | n |
|---|---|---|
| junior (0–2) | 0–2 years | 1,367 |
| mid (3–5) | 3–5 years | 2,918 |
| senior (6–10) | 6–10 years | 4,480 |
| veteran (10+) | 10+ years | 9,249 |

**AI Usage Flag (`UsesAI`)**

Boolean column derived from `df_pros['AISelect']`: `True` where response contains `'Yes'`, `False` otherwise. Assignment references `df_pros` (not the raw `df`) to maintain correct index alignment.

**YearsCode Dropped**

`YearsCode` (total years coding including pre-professional) was found to be highly correlated with `WorkExp` (professional experience only). Retaining both would introduce multicollinearity into the regression model. Dropped from `df_pros` before expansion.

**Multi-Select Column Expansion**

Four technology columns store semicolon-delimited responses. `sklearn.preprocessing.MultiLabelBinarizer` was applied to each, producing one binary indicator column per technology prefixed with the source column name to prevent collisions. The empty-string column produced by splitting null values was removed post-binarization.

---

### 1.5 Salary Outlier Cap

The raw `ConvertedCompYearly` distribution contains extreme self-reported values reaching ~$35M, clearly erroneous. A 1st–99th percentile cap was applied after confirming the extent of the problem from the salary distribution visualisation. `log_salary` was recalculated post-cap.

**Why the cap precedes log transformation:** The uncapped distribution visualisation was retained to document the problem and justify the decision. All subsequent analysis uses the capped dataset.

> **Note on scouting table timing:** The tech stack scouting tables (salary premium calculations) were computed on pre-cap `df_expanded`. Median-based metrics are robust to extreme outliers. Post-cap analysis confirms no material change to premium rankings or the ±15% feature selection threshold.

---

### 1.6 Education Level — Evaluated and Excluded

Education level was evaluated as a potential confounder. Median salary ranges from $61,878 (Secondary school) to $89,098 (Professional/PhD) — a $27k spread. However, AI adoption across education levels ranges only 76.9%–88.9%, with no systematic relationship between education level and AI adoption direction (Primary school, with a small sample, shows the highest adoption at 88.9%). The salary effect of education is dominated by Country and DevType. `EdLevelClean` was evaluated in a dedicated visualisation and is not carried into the model.

---

### 1.7 Region Encoding

`Country` was mapped to 10 geographic regions to reduce cardinality from ~177 countries to a model-manageable categorical feature while preserving the key salary gradients (North America vs South Asia represents a ~5:1 USD salary ratio):

| Region | Examples |
|---|---|
| North America | USA, Canada |
| Western Europe | UK, Germany, France, Netherlands, Switzerland, Italy, Spain |
| Eastern Europe | Poland, Ukraine, Czech Republic, Russia, Romania |
| Latin America | Brazil, Argentina, Mexico, Colombia |
| South Asia | India, Pakistan, Bangladesh |
| East Asia | China, Japan, South Korea |
| Southeast Asia | Indonesia, Philippines, Singapore, Vietnam |
| Middle East & North Africa | Israel, UAE, Turkey, Egypt, Saudi Arabia |
| Sub-Saharan Africa | South Africa, Nigeria, Kenya |
| Oceania | Australia, New Zealand |
| Other | Unmapped countries |

Unmapped countries were iteratively identified and added via a second `region_map.update()` block after inspecting the initial unmapped set.

---

### 1.8 Feature Selection — Tech Stack

Scouting analysis across all four technology categories. For each tool with ≥500 users, computed: median salary of users vs. non-users, salary premium over non-users (%), and AI adoption rate among users (%).

Tools with absolute salary premium >±15% were selected as model features. This threshold was chosen based on a natural break in the premium distribution; tools inside ±15% carry ambiguous salary signal relative to sample noise at this dataset scale.

**44 tools** met the threshold across Languages, Databases, Platforms, and Frameworks.

---

## Section 2 — Exploratory Data Analysis

### 2.1 Target Variable — Salary Distribution

Raw salary is heavily right-skewed with erroneous values reaching ~$35M. Log transformation (`log1p`) produces an approximately unimodal, right-skewed distribution with median log value 11.28 (equivalent to ~$78,890). The remaining right skew in the log-transformed data is within acceptable bounds for tree-based models.

**Decision:** `log_salary = log1p(ConvertedCompYearly)` is the regression target. When making predictions, reverse with `np.expm1(prediction)` to return dollar values.

---

### 2.2 Target Variable — AI Tool Usage (Classifier)

| Class | Count | Proportion |
|---|---|---|
| Uses AI | 14,705 | 81.6% |
| No AI | 3,309 | 18.4% |
| **Total** | **18,014** | |

**Significant class imbalance.** 81.6% of professional developers in this dataset currently use AI tools. A naive classifier always predicting "Uses AI" would achieve 81.6% accuracy while being completely uninformative. This imbalance must be addressed in modelling.

**Decision:** Apply `class_weight='balanced'` to all classifiers. This instructs the algorithm to penalise misclassifying the minority class (No AI) proportionally more, preventing the model from defaulting to always predicting the majority class.

---

### 2.3 Confound Investigation — AI Adoption by Experience Tier

| Tier | n | AI Adoption Rate |
|---|---|---|
| junior (0–2) | 1,367 | 86.8% |
| mid (3–5) | 2,918 | 85.5% |
| senior (6–10) | 4,480 | 83.4% |
| veteran (10+) | 9,249 | 78.8% |
| **Overall** | **18,014** | **81.6%** |

AI adoption shows a **clear decreasing trend** with experience. Junior developers adopt AI tools at the highest rate (86.8%), declining monotonically to veterans at 78.8% — an 8 percentage point spread.

**Finding — First arm of confound is present but inverted.** The confounding hypothesis assumes senior developers use AI more (driving a spurious AI–salary association). The data shows the opposite: junior developers use AI more aggressively. This inverts the assumed confound mechanism. If anything, the higher AI adoption among lower-earning juniors would suppress an apparent AI salary premium in a naive unadjusted comparison — not inflate it.

---

### 2.4 Confound Investigation — Salary by Experience Tier (Global)

| Tier | Median Salary (USD) |
|---|---|
| junior (0–2) | $27,217 |
| mid (3–5) | $44,978 |
| senior (6–10) | $74,249 |
| veteran (10+) | $100,000 |

Salary grows nearly **4x from junior to veteran**. Experience is a strong salary predictor in the corrected dataset. The previously flat chart in an earlier notebook run was caused by a data alignment bug (referencing `df['WorkExp']` instead of `df_pros['WorkExp']` when assigning `ExperienceTier`), which assigned experience tier values from the unfiltered raw dataset to the professional-developer-filtered dataframe, producing misaligned tier assignments.

**Finding — Second arm of confound is confirmed and strong.** Experience strongly predicts salary.

---

### 2.5 Confound Check — Salary by AI Usage Within Experience Tier (US Respondents)

US respondents are used here to isolate the experience–salary relationship from geographic salary compression (USD salary ratios of ~5:1 between North America and South Asia suppress the experience gradient in global data). The full model controls for this via `Region` encoding rather than filtering.

| Tier | No AI Median | Uses AI Median | Gap |
|---|---|---|---|
| junior (0–2) | $78,000 | $87,000 | **+$9,000** |
| mid (3–5) | $97,500 | $105,000 | **+$7,500** |
| senior (6–10) | $132,150 | $150,000 | **+$17,850** |
| veteran (10+) | $151,500 | $170,000 | **+$18,500** |

AI users earn more **in every single experience tier**. The premium grows substantially with seniority — a junior AI user earns ~$9k more than a non-AI-using peer; a veteran AI user earns ~$18.5k more.

**Finding — After controlling for experience, a positive AI salary premium persists and grows with seniority.** This is not explained by the confound mechanism: experience doesn't make developers more likely to use AI (see 2.3). The premium must therefore reflect something about what AI-using developers do differently, who employs them, or what kind of roles they hold — questions for the model to quantify.

---

### 2.6 Structural Confounders

**Country / Region**

US respondents earn a median ~$150,000 vs ~$30,000 for Ukraine and India — a 5:1 ratio in USD terms. Country is the single most powerful salary predictor in the dataset. Encoded as `Region` (10 geographic groups) for the model to learn regional salary baselines.

**Developer Type**

Engineering Managers ($135,000), Senior Executives ($122,527), and Cloud Infrastructure Engineers ($108,725) lead the salary distribution. DevType encodes seniority and specialisation more precisely than years of experience alone — a 2-year FAANG engineer can earn more than a 15-year agency developer. Included as an OHE feature.

**Industry**

Fintech ($95,575), Healthcare ($92,812), and Insurance ($91,534) lead across 12 industries. The ~$14k range from top to bottom industry is meaningful. Included as an OHE feature.

**Education Level**

AI adoption across all education levels falls within a 12pp range (76.9%–88.9%) with no meaningful correlation between education and AI adoption direction. Salary ranges $61,878–$89,098 — modest relative to Country and DevType. Education is excluded from the model.

---

### 2.7 Tech Stack — Salary Premium vs AI Adoption

**High-salary, high-AI adoption (top-right quadrant):** Enterprise cloud and data infrastructure tools — Snowflake, Datadog, Splunk, Databricks SQL, Terraform, AWS — show both elevated salary premiums (20–70%+) and above-average AI adoption. Their salary premium reflects the seniority and employer type of their typical users.

**Low-salary, high-AI adoption (top-left quadrant — the paradox group):** Modern web tools — Firebase, Vercel, NestJS, Supabase, Laravel — show the highest AI adoption rates (90–95%) while carrying negative salary premiums (−15% to −40%). Junior and mid-level developers on beginner-accessible stacks adopt AI most aggressively, consistent with the adoption-by-experience finding in 2.3.

**Central cluster:** Mass-adoption tools — JavaScript, Python, SQL, TypeScript, Docker — sit near 0% salary premium with ~82–85% AI adoption. Too universal to carry salary signal.

**Key observation:** High AI adoption does NOT cluster with high salary. The highest AI adopters are in the lower-salary quadrant. This is consistent with finding 2.3 — it is junior developers, not senior ones, who drive AI adoption rates upward.

---

### 2.8 Feature Selection Summary

44 tech tools selected via ±15% salary premium threshold (minimum 500 users).

**Positive-premium tools (enterprise/cloud/data stack):** Snowflake, Splunk, Datadog, Terraform, Databricks SQL, Ruby, Ruby on Rails, DynamoDB, Homebrew, Elixir, New Relic, AWS, Perl, Kubernetes, Bash/Shell, Go, Podman, Make, Cosmos DB, MSBuild, Swift, Prometheus, Ansible, BigQuery, Poetry, Groovy

**Negative-premium tools (web/beginner stack):** Firebase Realtime Database, Laravel, Dart, Cloud Firestore, PHP, Firebase, Composer, Django, Symfony, NestJS, Supabase, MongoDB, Nuxt.js, Vercel, WordPress, MySQL, MariaDB, Express

---

## EDA Summary & Modelling Rationale

**Research question:** Do developers who use AI tools earn more — or are senior devs just more likely to use AI?

---

### Hypothesis Rejection — Two-Arm Analysis

The confounding hypothesis requires two conditions to both hold:
1. Senior developers must use AI significantly more than junior developers
2. Senior developers must earn significantly more than junior developers

**Condition 1 — DOES NOT HOLD (inverted):** AI adoption *decreases* with experience. Junior developers adopt AI at 86.8%, veteran developers at 78.8%. The assumed confound mechanism — that AI adoption is driven by seniority — is not supported. If anything, the direction runs counter to the confound: higher AI adoption among lower-earning juniors would suppress a naive AI–salary premium, not create one.

**Condition 2 — CONFIRMED:** Salary grows nearly 4x from junior ($27,217) to veteran ($100,000). Experience strongly predicts salary.

**Conclusion on confound:** Since condition 1 does not hold, experience cannot be confounding the AI–salary relationship through the adoption channel. The "senior devs just use AI more" explanation is not supported by this data.

---

### The Actual Finding

Within every US experience tier, AI users earn more than non-AI-using peers. The premium ranges from +$7,500 (mid) to +$18,500 (veteran). This premium exists and persists after experience is controlled for. It is NOT explained by seniors using AI more (they don't). The model's job is to quantify this relationship and determine how much of the variance in salary `UsesAI` explains above and beyond experience, region, DevType, industry, and tech stack.

---

### Modelling Setup

**Regression target:** `log_salary` (log1p of capped `ConvertedCompYearly`)

**Classification target:** `UsesAI` (81.6% positive — **significant imbalance**)

**Class imbalance handling:** `class_weight='balanced'` required for all classifiers. Without this, classifiers will default to predicting "Uses AI" for every respondent and achieve a misleading 81.6% accuracy.

**Models to compare:** Linear Regression (baseline) → Random Forest → Gradient Boosting. Same approach for classifier: Logistic Regression → Random Forest → XGBoost.

**Primary control variable:** `WorkExp` (continuous) — included in all models so the `UsesAI` coefficient/SHAP value represents the marginal contribution of AI usage above and beyond experience.

**Secondary controls included as features:** `Region` (OHE from Country), `DevType` (OHE), `Industry` (OHE), `ExperienceTier_encoded` (ordinal).

**Tech stack features:** 44 tools via ±15% salary premium threshold, ≥500 users.

**K-fold validation:** 5-fold cross-validation (`KFold` for regression, `StratifiedKFold` for classification) used for model selection. Final model retrained on full training set; evaluated once on held-out test set.

---

*Stack Overflow Developer Survey 2025 | Project 3 | EDA Complete → Modelling*