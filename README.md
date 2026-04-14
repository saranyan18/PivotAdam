# PivotAdam: Memory-Efficient Full-Parameter Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

**PivotAdam** is a low-rank optimizer that enables **full-parameter fine-tuning** of billion-parameter models on consumer-grade hardware. By projecting gradients into dynamic, orthonormal subspaces, PivotAdam bypasses the massive memory "tax" of standard Adam optimizer states.

## Training on consumer grade hardware(with dedicated gpu)

Standard AdamW requires $2\times$ the model’s parameters in high-precision (float32) memory for optimizer states ($m$ and $v$). 

* **The Problem:** A 1.5B parameter model requires ~12GB of VRAM *just* for the optimizer states, making training impossible on 8GB GPUs.
* **The Solution:** PivotAdam reduces this overhead by **>90%**, allowing full-parameter updates of a **1.5B model using only ~7.3 GB of total VRAM**.

---

## 💡 Core Innovations

### 1. Randomized QR-Based Projection
Unlike GaLore or other methods that rely on computationally expensive Singular Value Decomposition (SVD), PivotAdam utilizes **Randomized QR Decomposition**. This provides a perfectly orthonormal basis ($P^\top P = \mathbf{I}$) with $O(Nd)$ complexity, significantly reducing per-step overhead.

### 2. Momentum Pivoting ($T = P^\top P_{new}$)
When the subspace basis refreshes, historical momentum is preserved through a geometric transition matrix $T$. This "pivots" the existing $m$ and $v$ buffers into the new coordinate system, ensuring trajectory continuity in the latent manifold.

### 3. Numerical Stability Fix
PivotAdam performs subspace transitions in **float32** before casting back to the model's native precision (e.g., `float16`). This prevents silent underflow of the variance ($v$) states during coordinate shifts—a critical fix for stable half-precision training.

---

## 📊 Benchmarks
*Tested on NVIDIA RTX 8GB (Dedicated VRAM)*

| Model | Parameters | Rank ($d$) | Optimizer | Peak VRAM | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen-2.5-1.5B** | 1.5B | 128 | AdamW | ~19.5 GB | ❌ **OOM** |
| **Qwen-2.5-1.5B** | 1.5B | 128 | **PivotAdam** | **7.33 GB** | ✅ **Success** |

### **Empirical Validation (Ablation Studies)**
* **Fake Signal Test:** Replacing the projected gradient with isotropic noise led to immediate loss stagnation, proving the projection successfully captures the true descent signal.
* **Chaos Test:** Forcing a basis change every step (`rotate_every=1`) resulted in 7x slower convergence, validating the necessity of subspace stability and the $T$-matrix rotation logic.

---

## 🛠️ Quick Start

### Installation
```bash
git clone [https://github.com/your-username/PivotAdam.git](https://github.com/your-username/PivotAdam.git)
cd PivotAdam
