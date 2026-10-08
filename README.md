# Visual Linear Algebra Engine: Live Interactive Dashboard

> **Semester 3 Applied Mathematics Project**  
> An interactive, 60 FPS real-time visual proof of fundamental Linear Algebra concepts (Units 1 & 2) using handwritten digit recognition as a concrete geometric substrate. Inspired by the minimalist aesthetic of **3Blue1Brown**.

---

## 1. Project Overview & Philosophy

Modern machine learning is often presented as a black box. This project removes all abstraction, implementing an end-to-end digit classification engine **purely in NumPy** and rendering all linear algebra transformations **simultaneously in real time on a unified interactive dashboard**.

### 1.1 Strict Curriculum Scope: Units 1 & 2 (Linear Algebra Only)
To maintain complete alignment with the Semester 3 syllabus, this application strictly implements:
- **Vector Spaces & Inner Products:** Converting spatial pixel grids into high-dimensional vectors $\mathbf{x} \in \mathbb{R}^{784}$ and computing projections.
- **Matrix Transformations:** Viewing dense neural layers as linear maps $\mathbf{W}: \mathbb{R}^n \to \mathbb{R}^m$.
- **Rank & Null Space (Rank-Nullity Theorem):** Proving that mapping $784 \to 10$ dimensions creates a 774-dimensional null space where structural noise $\mathbf{n}$ produces zero output change ($\mathbf{W}\mathbf{n} = \mathbf{0}$).
- **Ordinary Least Squares (Moore-Penrose Pseudoinverse):** Solving the closed-form normal equations $\mathbf{W}_{\text{LS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$ analytically and contrasting it with iterative gradient descent.
- **Singular Value Decomposition (SVD) & Principal Component Analysis (PCA):** Computing the principal eigenvectors of the covariance matrix to project $784$-dimensional user drawings onto a 2D geometric manifold in real time.

*(No inferential statistics, probability distributions, hypothesis tests, or confidence intervals from Units 3 & 4 are used.)*

---

## 2. Unified Live Dashboard Layout

Instead of stepping through disjointed slides, the dashboard renders the entire linear algebra pipeline **live simultaneously as you draw**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ VISUAL LINEAR ALGEBRA ENGINE (Units 1 & 2)                      ● Real-Time NumPy Inference │
├─────────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│ 1. INPUT VECTOR x ∈ ℝ⁷⁸⁴│ 2. SINGLE LINEAR MAP              │ 4. NULL SPACE COLLAPSE        │
│                         │    y = softmax(W_lin · x + b_lin) │    rank(W)=10, nullity=774    │
│ ┌─────────────────────┐ │    [ 0 ][ 1 ][ 2 ]...[ 9 ]        │    [ x ]  [ n ]  [ x + n ]    │
│ │ 28x28 Drawing Grid  │ │    10 Template Dot Products       │    ‖W(x+n) - Wx‖ ≈ 10⁻¹⁴      │
│ │ (Live Brush Stroke) │ ├───────────────────────────────────┼───────────────────────────────┤
│ └─────────────────────┘ │ 3. MULTI-LAYER PERCEPTRON (MLP)   │ 5. LEAST SQUARES (NORMAL EQ)  │
│                         │    x → h₁ = ReLU(W₁x + b₁)        │    W_LS = (XᵀX)⁻¹ Xᵀ Y        │
│ [Clear (C)] [Sample (R)]│    → h₂ = ReLU(W₂h₁ + b₂)         │    Bars: LS vs GD Weights     │
│                         │    → ŷ = softmax(W₃h₂ + b₃)       ├───────────────────────────────┤
│ Center of Mass Centered │    (784 → 16 → 16 → 10)           │ 6. 2D PCA & SVD SUBSPACE      │
│ [x̄] [c̄y, c̄x] Formula   │    Glowing Neurons & Synapses     │    1500 Reference Digit Plot  │
│                         │    Blue (+) / Orange (-) Weights  │    Live User Trajectory Point │
└─────────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

---

## 3. Real-Time Dashboard Panels Explained

### Panel 1: Input Vector & Discretization ($\mathbf{x} \in \mathbb{R}^{784}$)
- **Interactive Grid:** A high-contrast $28 \times 28$ matrix where each cell represents a scalar float $x_i \in [0.00, 1.00]$.
- **Pixel Inspector:** Hovering over any pixel immediately displays its exact vector index and intensity `x[i] = 0.85`.
- **Center of Mass Centering:** Dynamically calculates $(\bar{c}_y, \bar{c}_x)$ and shifts the drawing so that it aligns with MNIST standards.
- **Controls:** Clear canvas with `C`, generate sample digits with `R`, or watch Manim clips with `1`, `2`, `3`.

### Panel 2: Single Linear Map ($\mathbf{W}_{\text{lin}} \mathbf{x} + \mathbf{b}_{\text{lin}}$)
- **10 Visual Template Heatmaps:** Each $28 \times 28$ thumbnail renders a row $\mathbf{w}_k^T$ of the weight matrix $\mathbf{W}_{\text{lin}}$.
- **Live Inner Products:** Real-time vertical bars show the similarity score $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$.
- **Winning Template:** The highest scoring digit is highlighted with a bright yellow frame and softmax percentage.

### Panel 3: Multi-Layer Perceptron ($784 \to 16 \to 16 \to 10$)
- **Network Architecture:** Full vector-graph rendering connecting the 784-input bundle to Hidden Layer 1 (16 neurons), Hidden Layer 2 (16 neurons), and Output (10 classes).
- **ReLU Activations:** Neurons glow in white with opacity directly proportional to $\mathbf{a}_1 = \max(0, \mathbf{z}_1)$ and $\mathbf{a}_2 = \max(0, \mathbf{z}_2)$.
- **Synaptic Connections:** Weights are rendered as dynamic lines: **Cyan/Blue** for positive excitatory connections ($>0$) and **Orange/Red** for negative inhibitory connections ($<0$).

### Panel 4: Null Space & Rank-Nullity Theorem
- **Rank-Nullity Proof:** With $\mathbf{W}_{\text{lin}} \in \mathbb{R}^{10 \times 784}$ having $\text{rank} = 10$, the null space dimension is $784 - 10 = 774$.
- **Live Perturbation:** Generates a non-trivial null vector $\mathbf{n} = \mathbf{z} - \mathbf{W}^+(\mathbf{W}\mathbf{z})$.
- **Visual Invariance:** Shows $\mathbf{x}$, structural null noise $\mathbf{n}$, and the perturbed image $\mathbf{x} + \mathbf{n}$, while computing the real-time difference:
  $$\|\mathbf{W}(\mathbf{x} + \mathbf{n}) - \mathbf{W}\mathbf{x}\|_\infty \approx 10^{-14}$$

### Panel 5: Analytical Least Squares vs. Gradient Descent
- **Normal Equations:** Shows the closed-form pseudoinverse weights $\mathbf{B} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{X}^+ \mathbf{Y}$.
- **Live Response Comparison:** Real-time horizontal bar chart showing how the linear least-squares model compares against iterative gradient descent.

### Panel 6: SVD & 2D PCA Subspace Projection
- **2D Covariance Manifold:** Scatter plot of 1,500 reference dataset digits projected onto the top 2 right singular vectors ($\mathbf{v}_1, \mathbf{v}_2$) of the centered data matrix $\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$.
- **Live User Trajectory:** As you draw on the canvas, your live coordinate $(\mathbf{x} - \boldsymbol{\mu})\mathbf{V}_{1:2}^T$ glides smoothly across the 2D cluster manifold as a glowing yellow-and-white indicator ring.
- **Principal Vector Thumbnails:** Displays the top spatial eigenvectors.

---

## 4. Mathematical Deep Dive

### 4.1 Digit Centering via Center of Mass
$$\bar{c}_y = \frac{\sum_{i=0}^{27} \sum_{j=0}^{27} i \cdot I_{i,j}}{\sum_{i=0}^{27} \sum_{j=0}^{27} I_{i,j}}, \quad \bar{c}_x = \frac{\sum_{i=0}^{27} \sum_{j=0}^{27} j \cdot I_{i,j}}{\sum_{i=0}^{27} \sum_{j=0}^{27} I_{i,j}}$$
The image is shifted via circular roll by $(\text{round}(13.5 - \bar{c}_y), \text{round}(13.5 - \bar{c}_x))$.

### 4.2 Rank-Nullity Theorem & Null Space Invariance
For $\mathbf{W}_{\text{lin}} \in \mathbb{R}^{10 \times 784}$:
$$\text{dim}(\mathbb{R}^{784}) = \text{rank}(\mathbf{W}_{\text{lin}}) + \text{nullity}(\mathbf{W}_{\text{lin}}) \implies 784 = 10 + 774$$
For any vector $\mathbf{z} \in \mathbb{R}^{784}$, the null space projection is:
$$\mathbf{n} = \mathbf{z} - \mathbf{W}^+ (\mathbf{W} \mathbf{z})$$
Because $\mathbf{W}\mathbf{n} = \mathbf{0}$, it follows that:
$$\mathbf{W}(\mathbf{x} + \mathbf{n}) = \mathbf{W}\mathbf{x} + \mathbf{W}\mathbf{n} = \mathbf{W}\mathbf{x}$$

### 4.3 Closed-Form Ordinary Least Squares
$$\mathbf{B}^* = \arg\min_{\mathbf{B}} \|\tilde{\mathbf{X}} \mathbf{B} - \mathbf{Y}\|_F^2 = (\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^T \mathbf{Y} = \tilde{\mathbf{X}}^+ \mathbf{Y}$$

### 4.4 SVD and PCA 2D Manifold
$$\tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T \implies \mathbf{p} = (\mathbf{x} - \boldsymbol{\mu}) \mathbf{V}_{1:2}^T \in \mathbb{R}^2$$

---

## 5. Setup & Execution Guide

```bash
# 1. Install dependencies
pip install numpy flask scikit-learn

# 2. Train model & compute linear algebra artifacts (run once)
python train.py
python extras.py

# 3. Launch live web dashboard
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 6. Keyboard Shortcuts

| Key | Action |
|---|---|
| **`Mouse / Touch Drag`** | Draw digit on the $28 \times 28$ grid |
| **`C`** | Clear canvas |
| **`R`** | Generate sample digit preset |
| **`1` / `2` / `3`** | Play Manim animation clips (Pixels, MatVec, Null Space) |
| **`Escape`** | Close video player modal |
