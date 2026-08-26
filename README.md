# Week 6: Deep Learning & Applied Project — Sprint 1 (Baseline Pipeline)

**Welcome to Week 6 of the BinX Tech AI & Machine Learning Internship Program.**
This week begins **Phase 3 — Deep Learning & Applied Project**, a capstone-style phase structured as **4 one-week sprints** (Weeks 6–9). Sprint 1 (Week 6) focuses on establishing a robust, reproducible machine learning baseline pipeline on the Heart Disease dataset using standard tabular modeling techniques. Every neural network developed in subsequent sprints **must beat** the baseline metrics recorded this week.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Sprint 1 Kickoff & Baseline Model | [`sprint1_baseline.ipynb`](./Day1/sprint1_baseline.ipynb) | ✅ |
| 2 | Activations & Forward Pass (Neural Network Foundations) | [`Activations_the-Forward-Pass.ipynb`](./Day2/Activations_the-Forward-Pass.ipynb) | ✅ |
| 3 | Training Mechanics: Loss Curves, Learning Rate, Training Loop | [`Backpropagation_Gradient-Descent.ipynb`](./Day3/Backpropagation_Gradient-Descent.ipynb) | ✅ |
| 4 | Keras Neural Network: Compile/Fit/Evaluate, Dropout, Batch-Norm | [`Training_a_Network_in_Keras.ipynb`](./Day4/Training_a_Network_in_Keras.ipynb) | ✅ |
| 5 | Tuning: EarlyStopping, Metric Comparison vs Baseline, Retrospective | — | 🔲 |

---

## 📖 Summary by Day

### [Day 1 — Sprint 1 Kickoff & Baseline Model](./Day1/README.md)
Completed the **Sprint 1 planning** and built a **complete baseline ML pipeline** on the processed Heart Disease dataset (918 patients × 14 columns). Loaded the dataset, validated the schema (13 numeric features, 1 categorical, no missing values), and performed comprehensive EDA — statistical summaries, target distribution (55.3% positive / 44.7% negative), univariate analysis, IQR outlier detection, and correlation analysis. Assembled a **leakage-free preprocessing pipeline** using `ColumnTransformer` (median imputation + `StandardScaler` for numerics, most-frequent imputation + `OneHotEncoder` for categoricals) fitted on training data only. Trained a **Logistic Regression** baseline classifier and evaluated it with Accuracy, F1-Score, ROC-AUC, and a confusion matrix. These baseline scores establish the **absolute benchmark** that every neural network architecture in subsequent days must outperform. The notebook also defines the Sprint 1 backlog (5 tasks with acceptance criteria), sprint events timeline (Sprint Planning, Daily Stand-ups, Mentor Review, Sprint Review, Retrospective), and the full deliverables checklist for the sprint.

### [Day 2 — Activations & Forward Pass (Neural Network Foundations)](./Day2/README.md)
Completed a **hands-on lab** on neural network foundations — activation functions and the forward pass. Plotted and compared **ReLU, Sigmoid, Tanh, and Softmax** with their derivatives, explaining why non-linear activations are essential for deep networks. Selected **Sigmoid output + Binary Cross-Entropy loss** for the binary heart-disease classification task, with mathematical justification (the gradient simplifies to $\hat{y} - y$). Implemented a **complete forward pass from scratch in NumPy** using a 2-layer network (4 inputs → 3 hidden neurons with ReLU → 1 output neuron with Sigmoid), tracking tensor shapes at every step. Computed the BCE loss with numerical stability (clipping) and verified it analytically. Key takeaway: activations are not optional — without them, depth is meaningless; ReLU is the default for hidden layers; shape tracking is essential.

### [Day 3 — Training Mechanics: Loss Curves, Learning Rate, Training Loop](./Day3/README.md)
Completed a **hands-on lab** on the training loop, backpropagation, and learning rate effects. Described the **four-step training loop** (forward → loss → backprop → update) in a Markdown cell with a visual diagram. Implemented **backpropagation from scratch in NumPy** for the same 2-layer network from Day 2 (4 inputs → 3 hidden ReLU → 1 output Sigmoid), computing all gradients via the **chain rule**. Trained the network at **three learning rates** (0.5 too high, 0.0001 too low, 0.01 good) and plotted loss curves — demonstrating that too high causes divergence, too low causes slow training, and just right gives smooth convergence. Explained **what backpropagation computes** (gradients of the loss w.r.t. every weight) and **why the chain rule is involved** (the loss is a composition of nested functions). Opened a pull request for the mid-sprint Mentor Code & Notebook Review. Key insight: the learning rate is the single most important hyperparameter; start with Adam at lr=0.001.

### [Day 4 — Keras Neural Network: Compile/Fit/Evaluate, Dropout, Batch-Norm](./Day4/README.md)
Completed the **hands-on lab** on building and training a neural network using TensorFlow/Keras. Built a **baseline Sequential network** with two Dense hidden layers (64→32 neurons, ReLU) and a Sigmoid output layer for binary classification on the Heart Disease dataset. Compiled with Adam optimizer, binary cross-entropy loss, and accuracy metric; trained for 50 epochs with `validation_split=0.2`. Plotted training vs validation loss and accuracy curves, diagnosing **overfitting** from the widening gap between curves. Built an **enhanced model** incorporating `BatchNormalization()` after each Dense layer and `Dropout(0.3)` to regularize training. Overlaid the enhanced model's curves against the baseline to demonstrate stabilized training and reduced overfitting. Evaluated the best model on the unseen test set and compared results against the Day 1 Logistic Regression baseline (Accuracy: 0.8533, F1: 0.8657, ROC-AUC: 0.9159).

### Day 5 — Tuning: EarlyStopping, Metric Comparison vs Baseline, Retrospective
🔲 *Planned — not yet completed.*

---

## 🛠️ Skills & Tools Covered

| Skill | Day | Application |
|-------|:---:|-------------|
| **Sprint Planning** | 1 | Backlog, acceptance criteria, Definition of Done, sprint timeline |
| **Data Ingestion & Validation** | 1 | Schema check: shape, dtypes, missing values on 918×14 dataset |
| **Statistical Profiling** | 1 | Mean, median, std, skewness, kurtosis for all numeric features |
| **Target Distribution Analysis** | 1 | Class balance: 55.3% positive / 44.7% negative |
| **Univariate Analysis** | 1 | Histograms + KDE, box plots for distributions and outliers |
| **IQR Outlier Detection** | 1 | Interquartile range method for flagging extreme values |
| **Correlation Analysis** | 1 | Pearson correlation heatmap, top features vs target |
| **Leakage-Free Preprocessing** | 1 | `ColumnTransformer` + `Pipeline` (imputation, scaling, encoding) |
| **StandardScaler** | 1 | Feature normalization for distance/gradient-based models |
| **OneHotEncoder** | 1 | Categorical feature encoding |
| **Logistic Regression Baseline** | 1 | Binary classification with Accuracy, F1, ROC-AUC evaluation |
| **Confusion Matrix** | 1 | Visual heatmap of TP/TN/FP/FN |
| **ReLU Activation** | 2 | Hidden layer activation — zero for negatives, fast and sparse |
| **Sigmoid Activation** | 2 | Output activation for binary classification — probability output |
| **Tanh Activation** | 2 | Zero-centered hidden activation alternative |
| **Softmax Activation** | 2 | Multi-class output normalization |
| **Activation Derivatives** | 2 | Required for backpropagation gradient computation |
| **Binary Cross-Entropy** | 2 | Standard loss function for binary classification |
| **NumPy Forward Pass** | 2 | Manual implementation of neural network computation |
| **Shape Tracking** | 2 | Verifying tensor dimensions at each computation step |
| **Numerical Stability** | 2 | Clipping predictions to avoid log(0) errors |
| **Training Loop** | 3 | Forward → Loss → Backprop → Update — the heartbeat of every neural network |
| **Gradient Descent** | 3 | Moving weights opposite to the gradient to reduce loss |
| **Learning Rate** | 3 | Step size — most important hyperparameter; too high diverges, too low stalls |
| **Backpropagation** | 3 | Computing gradients efficiently via the chain rule |
| **Chain Rule** | 3 | Decomposing the gradient of a composition into local derivatives |
| **NumPy Backprop** | 3 | Manual implementation of forward + backward pass for a 2-layer network |
| **Loss Curves** | 3 | Visual diagnosis of training behavior — convergence, divergence, or slowness |
| **Mentor Review PR** | 3 | Mid-sprint pull request for structured code and notebook review |
| **Keras Sequential API** | 4 | Building layered neural networks by stacking Dense layers |
| **Dense Layers** | 4 | Fully connected layers with learnable weights and biases |
| **ReLU Activation** | 4 | Default hidden layer activation — fast, avoids vanishing gradient |
| **Sigmoid Activation** | 4 | Output activation for binary classification — probability output |
| **Adam Optimizer** | 4 | Adaptive learning rate optimizer — the default recommendation |
| **Binary Cross-Entropy** | 4 | Standard loss function paired with Sigmoid for binary classification |
| **Model Compilation** | 4 | `model.compile()` with optimizer, loss, and metrics |
| **Model Training** | 4 | `model.fit()` with validation split and epoch-based training |
| **Training History** | 4 | Storing and visualizing loss/accuracy curves for fit diagnosis |
| **Overfitting Diagnosis** | 4 | Identifying overfitting from diverging train/validation curves |
| **Batch Normalization** | 4 | Normalizing activations within mini-batches to stabilize training |
| **Dropout Regularization** | 4 | Randomly deactivating neurons to prevent co-adaptation |
| **Model Evaluation** | 4 | `model.evaluate()` on unseen test data |
| **Baseline Comparison** | 4 | Structured comparison of deep learning vs traditional ML baseline |

---

## 📁 Folder Structure

```
BinX_Week_06/
├── Day1/
│   ├── sprint1_baseline.ipynb
│   └── README.md
├── Day2/
│   ├── Activations_the-Forward-Pass.ipynb
│   └── README.md
├── Day3/
│   ├── Backpropagation_Gradient-Descent.ipynb
│   └── README.md
├── Day4/
│   ├── Training_a_Network_in_Keras.ipynb
│   └── README.md
├── Day5/                          # (planned)
└── README.md                      ← You are here
```

---

## 🚀 How to Run

1. **Clone the parent repository:**
   ```bash
   git clone --recurse-submodules https://github.com/adam-alAj/BinX-ML-Internship.git
   ```

2. **Navigate to Week 6:**
   ```bash
   cd BinX_ML_Internship/BinX_Week_06
   ```

3. **Activate the virtual environment** (located at the parent root):
   ```bash
   ..\.venv\Scripts\activate        # Windows
   source ../.venv/bin/activate     # Linux / macOS
   ```

4. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

5. **Launch Jupyter:**
   ```bash
   python -m jupyter notebook
   ```

---

## 🔗 Related

- [Root Repository README](../README.md) — Full internship overview and progress tracker
- [Week 5: Unsupervised Learning — Clustering & Dimensionality Reduction](../BinX_Week_05/README.md) — Previous module with clustering, PCA, t-SNE, and anomaly detection
