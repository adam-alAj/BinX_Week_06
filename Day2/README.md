# Day 2 — Hands-On Lab: Activations & the Forward Pass

**Week 6 · Phase 3 Sprint 1 — Deep Learning Intro**

---

## 📋 Table of Contents

| Step | Topic | Status |
|:----:|-------|:------:|
| 1 | Plotting Activation Functions & Their Derivatives | ✅ |
| 2 | Choosing Output Activation + Loss Function | ✅ |
| 3 | Forward Pass from Scratch in NumPy | ✅ |
| 4 | Summary & Key Takeaways | ✅ |

---

## 🎯 Learning Objectives

By the end of this notebook you will be able to:

1. **Explain why non-linear activations are essential** — without them, stacking layers collapses into a single linear transformation.
2. **Visualize and compare** ReLU, Sigmoid, Tanh, and Softmax, including their derivatives (critical for backpropagation on Day 3).
3. **Choose the correct output activation and loss function** for your task and justify the choice mathematically.
4. **Implement a forward pass from scratch** in NumPy, tracking tensor shapes at every step.
5. **Compute the loss** (Binary Cross-Entropy) and verify the result analytically.

### Why Activations Matter

A neural network without activation functions is just a chain of linear operations:

$$W_2(W_1 x + b_1) + b_2 = W'x + b'$$

No matter how many layers you stack, the result is always a **single linear model**. Activation functions introduce **non-linearity**, which is the single ingredient that lets deep networks learn complex, curved decision boundaries.

---

## 📊 Step 1 — Plotting Activation Functions & Their Derivatives

Plotted four common activation functions over the range $z \in [-6, 6]$ and their **derivatives**, because the derivative is what backpropagation (Day 3) uses to update weights.

| Function | Output Range | Typical Use |
|----------|-------------|-------------|
| **ReLU** | $[0, +\infty)$ | Hidden layers — the default, fast and effective |
| **Sigmoid** | $(0, 1)$ | Binary classification output — interpretable as probability |
| **Tanh** | $(-1, +1)$ | Hidden layers when zero-centered output helps |
| **Softmax** | $(0, 1)$, sums to 1 | Multi-class classification output — class probabilities |

### Key Observations

- **ReLU** is zero for all negative inputs → sparse activations, fast computation, but "dying ReLU" problem.
- **Sigmoid** saturates at 0 and 1 → gradients vanish for large $|z|$, slowing learning.
- **Tanh** is zero-centered → often trains faster than sigmoid in hidden layers.
- **Softmax** normalizes a vector into a probability distribution → only meaningful for the output layer.

---

## 🔧 Step 2 — Choosing Output Activation + Loss Function

For our **binary classification** task (heart disease present/absent):

- **Output Activation:** Sigmoid — outputs a value in $(0, 1)$ that can be interpreted as probability.
- **Loss Function:** Binary Cross-Entropy (BCE) — the standard loss for binary classification.

### Mathematical Justification

The BCE loss for a single sample is:

$$\mathcal{L} = -[y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y})]$$

When paired with Sigmoid output, the gradient simplifies to:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

This clean gradient makes BCE + Sigmoid a **matched pair** for binary classification.

---

## ⚙️ Step 3 — Forward Pass from Scratch in NumPy

Implemented a complete **2-layer neural network** forward pass:

```
Input (4 features) → Hidden Layer (3 neurons, ReLU) → Output (1 neuron, Sigmoid) → BCE Loss
```

### Architecture Details

| Layer | Shape | Activation |
|-------|-------|------------|
| Input $X$ | $(1, 4)$ | — |
| Hidden $Z^{[1]} = XW^{[1]} + b^{[1]}$ | $(1, 3)$ | ReLU |
| Output $Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}$ | $(1, 1)$ | Sigmoid |
| Prediction $\hat{y}$ | $(1, 1)$ | — |

### Weight Initialization

- **Hidden layer:** $W^{[1]} \in \mathbb{R}^{4 \times 3}$, initialized with `np.random.randn(4, 3) * 0.5`
- **Output layer:** $W^{[2]} \in \mathbb{R}^{3 \times 1}$, initialized with `np.random.randn(3, 1) * 0.5`
- **Biases:** initialized to zeros

### Sample Input

A single normalized patient record:
- $X = [0.54, 0.65, 0.47, 0.79]$ (age, trestbps, chol, thalach on [0,1] scale)
- $y = 1$ (heart disease present)

### Loss Computation

Used **Binary Cross-Entropy** with numerical stability via clipping:
```python
epsilon = 1e-15
y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
bce_loss = -(y_true * np.log(y_hat_clipped) + (1 - y_true) * np.log(1 - y_hat_clipped))
```

The result was verified analytically using the single-sample BCE formula.

---

## 💡 Step 4 — Summary & Key Takeaways

### What We Did

| Step | Outcome |
|------|---------|
| **Step 1** | Visualized ReLU, Sigmoid, Tanh, and Softmax with their derivatives — the building blocks of every neural network |
| **Step 2** | Selected **Sigmoid** (output activation) + **Binary Cross-Entropy** (loss) for our binary classification task, with mathematical justification |
| **Step 3** | Implemented a full forward pass in NumPy: $X \to Z^{[1]} \to A^{[1]} \to Z^{[2]} \to \hat{y}$, tracked shapes, and computed the BCE loss |
| **Step 4** | Verified the loss computation analytically |

### Key Insights

1. **Activations are not optional** — without them, depth is meaningless.
2. **ReLU is the default for hidden layers** — simple, fast, effective. Watch for dying ReLU (neurons stuck at 0).
3. **Output activation must match the task** — sigmoid for binary, softmax for multi-class, linear for regression.
4. **BCE + Sigmoid is a matched pair** — the gradient simplifies nicely: $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$.
5. **Shape tracking is essential** — most forward-pass bugs are shape mismatches.

---

## 🔜 Next: Day 3 — Backpropagation & Gradient Descent

Now that we can compute a prediction (forward pass) and measure how wrong it is (loss), the next step is to **compute gradients** (backpropagation) and **update the weights** (gradient descent) to reduce the loss.

---

## 🛠️ Skills & Tools Covered

| Skill | Application |
|-------|-------------|
| **ReLU** | Hidden layer activation — zero for negatives, fast and sparse |
| **Sigmoid** | Output activation for binary classification — probability output |
| **Tanh** | Zero-centered hidden activation alternative |
| **Softmax** | Multi-class output normalization |
| **Activation Derivatives** | Required for backpropagation gradient computation |
| **Binary Cross-Entropy** | Standard loss function for binary classification |
| **NumPy Forward Pass** | Manual implementation of neural network computation |
| **Shape Tracking** | Verifying tensor dimensions at each computation step |
| **Numerical Stability** | Clipping predictions to avoid log(0) errors |

---

*Part of [Week 6: Deep Learning & Applied Project — Sprint 1](../README.md)*
