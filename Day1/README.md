# Day 1 — Sprint 1 Kickoff & Baseline Model

Welcome to Day 1 of Week 6. This session marks the beginning of **Phase 3 — Deep Learning & Applied Project** and the first day of **Sprint 1**. We establish a robust, reproducible machine learning baseline pipeline on the Heart Disease dataset using standard tabular modeling techniques (Logistic Regression), whose metrics every neural network developed in subsequent sprints **must beat**.

---

## 🎯 Objective

Build a complete baseline ML pipeline that covers:

- Sprint 1 planning: confirm sprint goal, define backlog tasks with acceptance criteria, and establish the sprint timeline
- Data ingestion & schema validation on the processed Heart Disease dataset (918 patients × 14 columns)
- Comprehensive Exploratory Data Analysis (EDA): statistical summaries, target distribution, univariate analysis, outlier detection via IQR, and correlation analysis
- Leakage-free preprocessing pipeline assembly using Scikit-learn `ColumnTransformer` (imputation, scaling, encoding)
- Baseline model training with **Logistic Regression** and evaluation using Accuracy, F1-Score, and ROC-AUC
- Metric logging and benchmark definition for subsequent neural network models

---

## 📓 Notebook

- [sprint1_baseline.ipynb](./sprint1_baseline.ipynb)

---

## 📋 Sprint 1 Events & Timeline

Phase 3 runs across **4 one-week sprints**. Sprint 1 (Week 6) follows the standard sprint cycle:

| Event | When | Description |
|:------|:-----|:------------|
| **Sprint Planning** | Day 1 | Confirm the Sprint 1 goal, select backlog tasks (dataset, EDA, baseline), commit to the sprint goal. |
| **Daily Stand-up** | Daily | 3-minute update: what was completed, what is next, any blockers. |
| **Mentor Code & Notebook Review** | Day 3 | Mentor reviews notebook and code via GitHub PR with structured comments. |
| **Sprint Review** | Day 5 | Demo completed pipeline stages / model results. Incomplete tasks move to Sprint 2. |
| **Sprint Retrospective** | Day 5 | What went well, what to improve, and one specific action for Sprint 2. |

> **Notebook Workflow:** All work is committed to GitHub via **feature branches**. A pull request is opened for mentor review before merging to `main`.

---

## ✅ Key Tasks & Accomplishments

- **Step 1 — Data Ingestion & Schema Validation:** Loaded the processed **Heart Disease** dataset (`Data/processed/heart_disease_processed.csv`) — **918 rows × 14 columns**. Verified 13 numeric features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `num`, `target`) and 1 categorical feature (`country`). Confirmed **no missing values** across all columns.

- **Step 2 — Statistical Summary:** Profiled all 13 numeric features with `.describe()`, computing mean, median, std, skewness, and kurtosis. Key findings: `chol` is heavily right-skewed (skewness 1.61, kurtosis 7.03) indicating extreme cholesterol outliers (max 603); `fbs` is highly imbalanced (85% = 0); features span very different scales (`age` 28–77 vs `chol` 85–603) confirming the need for `StandardScaler`.

- **Step 3 — Target Distribution:** Assessed class balance — **target: 55.3% positive (disease) / 44.7% negative (no disease)** — a relatively balanced split that allows accuracy as a reasonable metric, though F1 and ROC-AUC remain important for a complete picture.

- **Step 4 — Univariate Analysis:** Generated histograms with KDE and box plots for all numeric features to visualize distributions and identify outliers. `chol`, `trestbps`, and `oldpeak` showed notable outlier presence.

- **Step 5 — Outlier Detection (IQR Method):** Applied the Interquartile Range (IQR) method to flag extreme values across features for investigation, ensuring anomalies are documented rather than silently dropped.

- **Step 6 — Correlation Analysis:** Produced a Pearson correlation heatmap and identified the top features most correlated with the target variable, guiding feature importance understanding for the baseline model.

- **Step 7 — Preprocessing Pipeline Assembly:** Built a leakage-free Scikit-learn `ColumnTransformer` pipeline — `SimpleImputer` + `StandardScaler` for numeric features, `SimpleImputer` + `OneHotEncoder` for categorical features — fitted **on the training set only** to prevent data leakage.

- **Step 8 — Baseline Model Training (Logistic Regression):** Trained a `LogisticRegression(random_state=42)` classifier on the preprocessed training data. Evaluated on the held-out test set using **Accuracy**, **F1-Score**, and **ROC-AUC**. Generated a full `classification_report` and confusion matrix heatmap. These scores establish the **absolute benchmark** that every neural network in subsequent days must outperform.

---

## 🛠️ Skills Covered

- Sprint planning with backlog, acceptance criteria, and Definition of Done
- Data ingestion and schema validation (shape, dtypes, missing values)
- Statistical profiling: mean, median, std, skewness, kurtosis
- Target distribution & class balance assessment
- Univariate analysis with histograms + KDE and box plots
- IQR-based outlier detection
- Pearson correlation heatmap for feature-target relationships
- Leakage-free preprocessing with `ColumnTransformer` (imputation, scaling, encoding)
- `StandardScaler` for feature normalization
- `OneHotEncoder` for categorical feature encoding
- Baseline model training with `LogisticRegression`
- Model evaluation: Accuracy, F1-Score, ROC-AUC, confusion matrix
- Benchmark definition for comparative model evaluation

---

## 🔗 Related

- [Week 6 Overview](../README.md)
- [Root Repository README](../../README.md)
