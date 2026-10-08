# AGENTS.md: Developer & AI Maintainer Guide

This document defines the architectural guidelines, mathematical specifications, live dashboard requirements, and code design rules for AI collaborators maintaining or modifying this repository.

---

## 1. Core Directives & Academic Constraints

### 1.1 Strict Curriculum Scope: Units 1 & 2 Only (Linear Algebra)
- **Included Topics:** Vector spaces, matrix transformations, matrix-vector multiplication, dot products, linear combinations, rank and nullity (Rank-Nullity Theorem), null space computations, eigenvalues and eigenvectors, singular value decomposition (SVD), principal component analysis (PCA), and least squares approximations via normal equations and Moore-Penrose pseudoinverse.
- **Strictly Excluded Topics:** Probability distributions, random variables, hypothesis testing, confidence intervals, p-values, or inferential statistics (Units 3 & 4). Softmax is treated strictly as an exponential normalization vector operator that rescales output logits to sum to $1$.

### 1.2 Pure NumPy Backend Rule
- Neural network forward passes, layer-by-layer activations, weight updates, matrix inversions, projections, and eigenvalue/SVD decompositions must be written purely using standard `numpy` primitives (`@`, `np.linalg.svd`, `np.linalg.pinv`, `np.linalg.matrix_rank`, `np.maximum`).
- **No high-level ML frameworks** (PyTorch, TensorFlow, Keras, etc.) are allowed during runtime inference or forward propagation. Scikit-learn is restricted solely to offline dataset ingestion (`fetch_openml`) in `train.py` and `extras.py`.

### 1.3 Live Interactive Dashboard Architecture (No Slide Shows)
- **Principle:** The frontend must never operate as a disjointed "slide show" or passive result display. It is an interactive, unified **Live Dashboard** rendering all mathematical transformations simultaneously at 60 FPS in real time.
- **Real-Time Dynamic Execution Flow:**
  - As the user draws on the $28 \times 28$ grid, the input vector $\mathbf{x} \in \mathbb{R}^{784}$ immediately flows through the linear algebra pipeline:
    1. **Center of Mass Centering:** Dynamically shifting the pixel matrix to $(13.5, 13.5)$.
    2. **Linear Map ($\mathbf{W}_{\text{lin}}\mathbf{x} + \mathbf{b}_{\text{lin}}$):** 10 template heatmaps and similarity inner products update live.
    3. **Deep Multi-Layer Network ($784 \to 16 \to 16 \to 10$):** Synaptic connection brightness (blue for $>0$, orange for $<0$), neuron ReLU glow ($\mathbf{a}_1, \mathbf{a}_2$), and output softmax probabilities update live.
    4. **Null Space Perturbation:** Live generation of null vector $\mathbf{n} \in \text{Null}(\mathbf{W}_{\text{lin}})$ proving $\mathbf{W}(\mathbf{x}+\mathbf{n}) = \mathbf{W}\mathbf{x}$ with difference $\approx 10^{-14}$.
    5. **Least Squares Normal Equations:** Real-time bar chart showing $\mathbf{W}_{\text{LS}}\mathbf{x}$ vs. $\mathbf{W}_{\text{lin}}\mathbf{x}$.
    6. **SVD & 2D PCA Subspace:** Live coordinate $(\mathbf{x}-\boldsymbol{\mu})\mathbf{V}_{1:2}^T$ moving continuously across the 1,500 reference digit cluster plot in real time.

### 1.4 3Blue1Brown Minimalist Aesthetic Guidelines
- **Background:** Strict pitch-black (`#000000`).
- **Typography:** Georgia / serif mathematical typesetting, clean monospaced indicators, subtle muted gray guides (`#666666`, `#cfcfcf`).
- **Color Coding:**
  - Positive weights / dot products / activations: Cyan/Blue (`#58C4DD` / `rgb(88, 196, 221)`).
  - Negative weights / inhibitions: Orange/Red (`#FC6255` / `rgb(252, 98, 85)`).
  - Maximum Activation / Winner / Highlights: Yellow (`#FFFF00`).
  - Active Neurons: White fill proportional to activation magnitude $[0, 1]$.
- **Visual Design:** Pure 2D vector canvas rendering. Absolutely no generic web styling, glossy gradients, shadows, cards, or glassmorphism.

---

## 2. Mathematical Formalism & Detailed Pipeline

### 2.1 The Input Vector $\mathbf{x} \in \mathbb{R}^{784}$
A user draws on a $28 \times 28$ continuous canvas grid.
1. The canvas pixels are normalized to $[0, 1]$.
2. The digit is centered according to its Center of Mass $(\bar{c}_y, \bar{c}_x)$:
   $$\bar{c}_y = \frac{\sum_{i,j} i \cdot I_{i,j}}{\sum_{i,j} I_{i,j}}, \quad \bar{c}_x = \frac{\sum_{i,j} j \cdot I_{i,j}}{\sum_{i,j} I_{i,j}}$$
3. The image is translated via circular roll such that $(\bar{c}_y, \bar{c}_x)$ moves to $(13.5, 13.5)$.
4. The matrix is flattened into vector $\mathbf{x} = \begin{bmatrix} x_0 & x_1 & \cdots & x_{783} \end{bmatrix}^T \in \mathbb{R}^{784}$.

### 2.2 Model 1: Single Linear Transformation ($784 \to 10$)
- **Weight Matrix:** $\mathbf{W}_{\text{lin}} \in \mathbb{R}^{10 \times 784}$, **Bias Vector:** $\mathbf{b}_{\text{lin}} \in \mathbb{R}^{10}$.
- **Linear Transformation:**
  $$\mathbf{z}_{\text{lin}} = \mathbf{W}_{\text{lin}} \mathbf{x} + \mathbf{b}_{\text{lin}}$$
  Each component $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$ represents the inner product between the input vector $\mathbf{x}$ and the learned visual template row $\mathbf{w}_k^T$.
- **Softmax Normalization:**
  $$y_k = \frac{e^{z_k - \max(\mathbf{z})}}{\sum_{j=0}^{9} e^{z_j - \max(\mathbf{z})}}$$

### 2.3 Model 2: Multi-Layer Perceptron ($784 \to 16 \to 16 \to 10$)
1. **Layer 1:**
   $$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{W}_1 \in \mathbb{R}^{16 \times 784}, \; \mathbf{b}_1 \in \mathbb{R}^{16}$$
   $$\mathbf{a}_1 = \max(0, \mathbf{z}_1) \in \mathbb{R}^{16} \quad (\text{ReLU Activation})$$
2. **Layer 2:**
   $$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{16 \times 16}, \; \mathbf{b}_2 \in \mathbb{R}^{16}$$
   $$\mathbf{a}_2 = \max(0, \mathbf{z}_2) \in \mathbb{R}^{16}$$
3. **Output Layer:**
   $$\mathbf{z}_3 = \mathbf{W}_3 \mathbf{a}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{10 \times 16}, \; \mathbf{b}_3 \in \mathbb{R}^{10}$$
   $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}_3) \in \mathbb{R}^{10}$$

### 2.4 Null Space and Dimensionality Collapse
- **Dimension Reduction:** $\mathbf{W}_{\text{lin}}: \mathbb{R}^{784} \to \mathbb{R}^{10}$.
- **Rank-Nullity Theorem:**
  $$\text{dim}(\text{Null}(\mathbf{W}_{\text{lin}})) = 784 - \text{rank}(\mathbf{W}_{\text{lin}}) = 784 - 10 = 774$$
- **Null Vector Construction:**
  $$\mathbf{n} = \mathbf{z} - \mathbf{W}_{\text{lin}}^+ (\mathbf{W}_{\text{lin}} \mathbf{z})$$
  where $\mathbf{W}_{\text{lin}}^+ = \mathbf{W}^T (\mathbf{W} \mathbf{W}^T)^{-1}$ is the Moore-Penrose pseudoinverse.
  $$\mathbf{W}_{\text{lin}} (\mathbf{x} + \mathbf{n}) = \mathbf{W}_{\text{lin}} \mathbf{x} + \mathbf{0} = \mathbf{W}_{\text{lin}} \mathbf{x}$$

### 2.5 Ordinary Least Squares via Moore-Penrose Pseudoinverse
- **Closed-Form Normal Equations:**
  $$\tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \mathbf{B} = \tilde{\mathbf{X}}^T \mathbf{Y} \implies \mathbf{B} = \tilde{\mathbf{X}}^+ \mathbf{Y} = (\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^T \mathbf{Y}$$

### 2.6 SVD & Principal Component Analysis (PCA)
- **Data Matrix:** Mean-centered data $\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu} \in \mathbb{R}^{N \times 784}$.
- **Singular Value Decomposition (SVD):** $\tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$.
- **2D Projection of Live Drawing:** $\mathbf{p} = (\mathbf{x} - \boldsymbol{\mu}) \mathbf{V}_{1:2}^T \in \mathbb{R}^2$.

---

## 3. Backend & Frontend Architecture

### 3.1 `app.py`
- Serves static assets (`index.html`, Manim concept clips).
- **Endpoints:**
  - `GET /model`: Returns network architecture matrices, least squares weights, PCA points, rank, and accuracies for instantaneous 60 FPS client rendering.
  - `POST /predict`: Accepts raw 784-element float array `px`, centers the digit, evaluates all layer activations, null space vector `n`, least squares prediction `ls`, and PCA projection `pc`.

### 3.2 `static/index.html` (Unified Live Dashboard)
- **Column 1 (Left):** $28 \times 28$ Interactive Drawing Pad, hover pixel inspector, Center of Mass preview, Clear (`C`), Random Sample (`R`), and Manim clip buttons (`1`, `2`, `3`).
- **Column 2 (Center):**
  - *Top:* Single Linear Map with 10 Template heatmaps and inner product bars.
  - *Bottom:* Deep MLP ($784 \to 16 \to 16 \to 10$) with glowing neurons, blue/orange synaptic connections, and softmax outputs.
- **Column 3 (Right):**
  - *Top:* Live Null Space preview ($\mathbf{x}$, $\mathbf{n}$, $\mathbf{x}+\mathbf{n}$) and numerical proof.
  - *Middle:* Least Squares vs Gradient Descent bar chart.
  - *Bottom:* Live 2D PCA scatter plot with live user trajectory point and eigenvector thumbnails.
