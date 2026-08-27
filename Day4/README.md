# Day 4 — Keras Neural Network: Compile/Fit/Evaluate, Dropout, Batch-Norm

Welcome to Day 4 of Week 6. This session covers building, training, regularizing, and evaluating a neural network using TensorFlow/Keras on the Heart Disease binary classification dataset.

---

## 🎯 Objective

Build a complete Keras deep learning pipeline covering:

- Design a baseline Sequential network with correct output activation (Sigmoid) and loss function (Binary Cross-Entropy) for binary classification
- Compile the model using the Adam optimizer and train with `model.fit()` for 50 epochs with validation monitoring
- Plot and diagnose training/validation loss and accuracy curves to identify overfitting
- Apply Batch Normalization and Dropout regularization to stabilize training and improve generalization
- Evaluate the best model on the unseen test set and compare against the Day 1 Logistic Regression baseline

---

## 📓 Notebook

- [Training_a_Network_in_Keras.ipynb](./Training_a_Network_in_Keras.ipynb)

---

## ✅ Key Tasks & Accomplishments

- **Step 1 — Architecture Design:** Built a baseline Keras `Sequential` network with two Dense hidden layers (64 → 32 neurons, ReLU activation) and a Sigmoid output layer for binary classification. Justified the Sigmoid activation as the mathematically correct pairing with Binary Cross-Entropy loss for probability outputs.

- **Step 2 — Model Compilation & Training:** Compiled the network with Adam optimizer (lr=0.001), `binary_crossentropy` loss, and `accuracy` metric. Trained for 50 epochs with `validation_split=0.2` (20% of training data held out for monitoring). Saved the training history in a `history` object.

- **Step 3 — History Plotting & Fit Diagnosis:** Generated Training vs Validation Loss and Accuracy curves across all epochs. Diagnosed the model as exhibiting **overfitting** based on the widening gap between training and validation loss curves.

- **Step 4 — Regularization & Architectural Enhancements:** Built an enhanced model incorporating `BatchNormalization()` after each Dense layer and `Dropout(0.3)` to prevent neuron co-adaptation. Trained the enhanced model under the same setup and overlaid the new loss/accuracy curves against the baseline run to demonstrate stabilized training and reduced overfitting.

- **Step 5 — Test Evaluation & Baseline Comparison:** Evaluated the best version of the trained network (enhanced model) on the unseen test dataset using `model.evaluate()`. Created a comparative Markdown table with dynamic code output comparing the final test scores against the Day 1 Logistic Regression baseline (Accuracy: 0.8533, F1: 0.8657, ROC-AUC: 0.9159). The enhanced model's curves were overlaid against the baseline run to visually demonstrate stabilized training and reduced overfitting.

---

## 🛠️ Skills Covered

| Skill | Application |
|:------|:------------|
| **Keras Sequential API** | Building layered neural networks by stacking Dense layers |
| **Dense Layers** | Fully connected layers with learnable weights |
| **ReLU Activation** | Default hidden layer activation — fast, avoids vanishing gradient |
| **Sigmoid Activation** | Output activation for binary classification — probability output ∈ [0, 1] |
| **Binary Cross-Entropy** | Standard loss function paired with Sigmoid for binary classification |
| **Adam Optimizer** | Adaptive learning rate optimizer — the default recommendation |
| **Model Compilation** | `model.compile()` with optimizer, loss, and metrics |
| **Model Training** | `model.fit()` with validation split and epoch-based training |
| **Training History** | Storing and visualizing loss/accuracy curves for fit diagnosis |
| **Overfitting Diagnosis** | Identifying overfitting from diverging train/validation curves |
| **Batch Normalization** | Normalizing activations within mini-batches to stabilize training |
| **Dropout Regularization** | Randomly deactivating neurons to prevent co-adaptation |
| **Model Evaluation** | `model.evaluate()` on unseen test data |
| **Baseline Comparison** | Structured comparison of deep learning vs traditional ML baseline |

---

## 🛠️ Tools Used

`scikit-learn` (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, metrics) · TensorFlow/Keras (`Sequential`, `Dense`, `BatchNormalization`, `Dropout`) · NumPy · Pandas · Matplotlib

---

## 🔗 Related

- [Week 6 Overview](../README.md)
- [Root Repository README](../../README.md)
- [Day 1 — Baseline Model](../Day1/README.md)
- [Day 2 — Activations & Forward Pass](../Day2/README.md)
- [Day 3 — Backpropagation & Gradient Descent](../Day3/README.md)
