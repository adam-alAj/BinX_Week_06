# Day 5 — Tuning, Evaluation & Sprint Review — Sprint 1 Close-Out

Welcome to Day 5 of Week 6. This session is the **final close-out for Sprint 1**, covering systematic hyperparameter tuning, callbacks (EarlyStopping & ModelCheckpoint), final model evaluation, and the sprint retrospective.

---

## 🎯 Objective

- Systematically tune a neural network using disciplined **one-variable-at-a-time** experiments
- Apply **EarlyStopping** to prevent overfitting and restore the best model weights
- Use **ModelCheckpoint** to persist the best-performing model during training
- **Evaluate the final tuned model** on the held-out test set and compare against the Day 1 Logistic Regression baseline and the Day 4 enhanced model
- Assemble **Sprint 1 evidence** — acceptance criteria, sprint review, and retrospective

### Context

This notebook builds directly on:

- **Day 1** — Logistic Regression baseline (Accuracy: 0.8533, F1: 0.8657, ROC-AUC: 0.9159)
- **Day 4** — Initial neural network with BatchNorm + Dropout (Accuracy: 0.8261, F1: 0.8447, ROC-AUC: 0.8991)

The Day 4 neural network did not beat the baseline. Today's goal was to systematically tune the network to find a configuration that could match or exceed baseline performance.

---

## 📓 Notebook

- [Evaluation_Sprint-Review.ipynb](./Evaluation_Sprint-Review.ipynb)

---

## ✅ Key Tasks & Accomplishments

- **Baseline Review:** Documented the Day 4 configuration and results explicitly. Recorded baseline metrics for comparison across all experiments.

- **Tuning Strategy:** Applied disciplined one-variable-at-a-time tuning across four hyperparameters in priority order: Learning Rate → Architecture → Dropout Rate → Batch Size. All experiments used 50 epochs and validation_split=0.2 for fair comparison.

- **Experiment 1 — Learning Rate:** Tested 5 learning rates [0.0001, 0.0005, 0.001, 0.005, 0.01] with fixed architecture [64, 32], dropout=0.3, batch_size=32. Selected **lr=0.0005** based on best validation loss (0.4229).

- **Experiment 2 — Network Architecture:** Tested 4 architectures with the best learning rate: [32], [64, 32], [128, 64], [128, 64, 32]. Selected **[128, 64, 32]** as the best configuration.

- **Experiment 3 — Dropout Rate:** Tested dropout values [0.0, 0.2, 0.3, 0.4, 0.5] with the best architecture and learning rate. Selected **dropout=0.2** as the best value.

- **Experiment 4 — Batch Size:** Tested batch sizes [16, 32, 64, 128] with the best configuration from previous experiments. Selected **batch_size=32**.

- **Consolidated Experiment Log:** Recorded all 18+ experiments in a single summary table for Sprint Review evidence.

- **EarlyStopping Implementation:** Applied `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` to prevent overfitting and automatically restore the best model weights.

- **ModelCheckpoint Implementation:** Applied `ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)` to persist the best model during training.

- **Final Model Training & Evaluation:** Trained the final tuned model with both callbacks and evaluated on the held-out test set.

- **Model Comparison:** Compared the final tuned neural network against both the Day 1 Logistic Regression baseline and the Day 4 enhanced model:

| Model | Accuracy | F1-Score | ROC-AUC |
|:------|:---------|:---------|:--------|
| Day 1 Baseline (Logistic Regression) | 0.8533 | 0.8657 | 0.9159 |
| Day 4 Neural Network (Enhanced) | 0.8261 | 0.8447 | 0.8991 |
| Day 5 Tuned Neural Network | 0.8370 | 0.8485 | 0.9076 |

- **Sprint 1 Acceptance Criteria:** Evaluated against 10 acceptance criteria — all 10/10 PASSED.

- **Sprint 1 Retrospective:** Documented what went well (baseline-first approach, leakage-free preprocessing, systematic tuning, callbacks, reproducibility) and what could be improved (NN did not beat baseline, limited feature engineering, no cross-validation, narrow tuning space, no learning rate scheduling).

- **Key Insight:** The neural network did not beat the Logistic Regression baseline on any metric after extensive tuning. This demonstrates that more complex models do not always outperform simpler baselines, especially on small tabular datasets (918 samples, 15 features).

---

## 🛠️ Skills Covered

| Skill | Application |
|:------|:------------|
| **One-Variable-at-a-Time Tuning** | Disciplined hyperparameter search — change one variable, record impact |
| **Learning Rate Tuning** | Testing multiple learning rates to find optimal convergence speed |
| **Architecture Tuning** | Testing different hidden layer configurations to find optimal capacity |
| **Dropout Tuning** | Testing different dropout rates to find optimal regularization |
| **Batch Size Tuning** | Testing different batch sizes to find optimal gradient noise |
| **EarlyStopping** | Preventing overfitting with patience-based early termination |
| **ModelCheckpoint** | Persisting the best model weights during training |
| **Experiment Logging** | Consolidating all experiments in a single summary table |
| **Model Comparison** | Structured comparison of baseline, enhanced, and tuned models |
| **Sprint Review** | Documenting completion status, evidence, and acceptance criteria |
| **Sprint Retrospective** | Reflecting on what went well and what could be improved |

---

## 🛠️ Tools Used

`scikit-learn` (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, metrics) · TensorFlow/Keras (`Sequential`, `Dense`, `BatchNormalization`, `Dropout`, `EarlyStopping`, `ModelCheckpoint`) · NumPy · Pandas · Matplotlib

---

## 🔗 Related

- [Week 6 Overview](../README.md)
- [Root Repository README](../../README.md)
- [Day 1 — Baseline Model](../Day1/README.md)
- [Day 2 — Activations & Forward Pass](../Day2/README.md)
- [Day 3 — Backpropagation & Gradient Descent](../Day3/README.md)
- [Day 4 — Keras Neural Network](../Day4/README.md)
