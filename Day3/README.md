# Day 3 — Hands-On Lab: Understanding Training + Mentor Review

**Week 6 · Phase 3 Sprint 1 — Deep Learning Intro**

---

##  Table of Contents

| Step | Topic | Status |
|:----:|-------|:------:|
| 1 | The Four-Step Training Loop (Described in Markdown) | ✅ |
| 2 | Learning Rate Experiments & Loss Curve Plots | ✅ |
| 3 | Backpropagation & the Chain Rule Explained | ✅ |

---

##  Learning Objectives

By the end of this notebook you will be able to:

1. **Describe the four-step training loop** of a neural network: forward → loss → backprop → update.
2. **Explain gradient descent** and the critical role of the **learning rate**.
3. **Implement backpropagation from scratch** in NumPy for the same 2-layer network from Day 2.
4. **Experiment with three learning rates** (too high, too low, good) and plot loss curves to see the effect.
5. **Explain what backpropagation computes** and why the **chain rule** is involved.

---

##  Step 1 — The Four-Step Training Loop

Described the four-step training loop in a Markdown cell:

| Step | Operation | What Happens |
|:----:|-----------|-------------|
| ① | **Forward Pass** | Data flows forward through the network: each layer computes Z = X·W + b, then A = g(Z) |
| ② | **Compute Loss** | Compare prediction ŷ to true label y using BCE loss |
| ③ | **Backpropagation** | Compute gradients of loss w.r.t. every weight using the chain rule |
| ④ | **Update Weights** | Move each weight opposite to its gradient: W = W − α · ∂L/∂W |

Repeat for many epochs → random weights become accurate weights.

---

##  Step 2 — Learning Rate Experiments

Trained the **same 2-layer network from Day 2** (4 inputs → 3 hidden ReLU → 1 output Sigmoid) at three learning rates:

| Learning Rate | Category | Behavior |
|:---:|:---:|---|
| **0.5** | Too High | Loss oscillates / diverges — overshoots the minimum |
| **0.0001** | Too Low | Loss decreases very slowly — tiny steps |
| **0.01** | Good | Smooth, efficient convergence to low loss |

### Loss Curves

Plotted individual and combined loss curves for all three learning rates. Key observation: the learning rate is the single most important hyperparameter — start with Adam at lr=0.001 as a reliable default.

---

##  Step 3 — Backpropagation & the Chain Rule

### What Backpropagation Computes

Backpropagation computes the **gradient** (partial derivative) of the loss with respect to **every weight and bias** in the network. It answers: *"If I slightly change this weight, how does the loss change?"*

For our 2-layer network:

- ∂L/∂W² — gradient for output layer weights
- ∂L/∂b² — gradient for output bias
- ∂L/∂W¹ — gradient for hidden layer weights
- ∂L/∂b¹ — gradient for hidden bias

### Why the Chain Rule

The loss is a **composition of nested functions**: the output is the result of multiple operations (linear → activation → linear → activation → loss). The chain rule decomposes this composition into manageable local derivatives:

$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{\partial \mathcal{L}}{\partial A^{[2]}} \cdot \frac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}}$$

Backpropagation is the **recursive application of the chain rule**, starting from the loss and moving backward.

### Code Mirror

```python
# Output layer — chain rule from loss to W2:
dZ2 = A2 - y                    # ∂L/∂Z2  (simplified for BCE + Sigmoid)
dW2 = A1.T.dot(dZ2) / m        # ∂L/∂W2 = A1.T · dZ2

# Hidden layer — chain rule continues backward:
dA1 = dZ2.dot(W2.T)            # ∂L/∂A1 = dZ2 · W2.T
dZ1 = dA1 * relu_derivative(Z1) # ∂L/∂Z1 = dA1 · g'(Z1)
dW1 = X.T.dot(dZ1) / m         # ∂L/∂W1 = X.T · dZ1
```

---


## 💡 Summary & Key Takeaways

### What We Did

| Step | Outcome |
|------|---------|
| **Step 1** | Described the four-step training loop: forward → loss → backprop → update |
| **Step 2** | Trained the 2-layer network at 3 learning rates and plotted loss curves |
| **Step 3** | Explained backpropagation as gradient computation via the chain rule |
| **Step 4** | Documented the pull request step for Review |

### Key Insights

1. **The training loop has four steps** — forward, loss, backprop, update. Every neural network follows this same loop.
2. **Backpropagation computes gradients** — it answers *"how much does each weight contribute to the error?"* via the chain rule.
3. **The chain rule is essential** because the loss is a composition of many functions.
4. **The learning rate is the most important hyperparameter** — too high causes divergence, too low causes slow training.
5. **BCE + Sigmoid simplifies the output gradient** to ŷ − y — clean and numerically stable.
6. **Frameworks compute backprop automatically** — PyTorch's autograd, TensorFlow's GradientTape — but understanding what they do is essential.

---

##  Next: Day 4 — Building & Training a Network in Keras

Now that we understand the training loop from scratch, we will use **TensorFlow/Keras** to build, train, and evaluate a neural network — with batch normalization and dropout.

---

## 🛠️ Skills & Tools Covered

| Skill | Application |
|-------|-------------|
| **Training Loop** | Forward → Loss → Backprop → Update — the heartbeat of every neural network |
| **Gradient Descent** | Moving weights opposite to the gradient to reduce loss |
| **Learning Rate** | Step size — most important hyperparameter |
| **Backpropagation** | Computing gradients efficiently via the chain rule |
| **Chain Rule** | Decomposing the gradient of a composition into local derivatives |
| **NumPy Backprop** | Manual implementation of forward + backward pass |
| **Loss Curves** | Visual diagnosis of training behavior |
| **Binary Cross-Entropy** | Loss function with simplified gradient for Sigmoid output |
| **Weight Initialization** | Random init with scaling for stable gradients |
| **Mentor Review PR** | Mid-sprint pull request for structured review |

---

*Part of [Week 6: Deep Learning & Applied Project — Sprint 1](../README.md)*
