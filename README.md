# SAWL-Net: Stability-Aware & Reproducible Histopathology Classification 🔬🧬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DRDO-Inspired](https://img.shields.io/badge/Research-DRDO%20Context-red)]()

## 🚀 The Mission
Medical AI often fails in production due to numerical instability and "black-box" decisions. **SAWL-Net** (Statistical-Wave Attention Lightweight Network) is a resource-efficient architecture that solves this. By fusing **Pearson/Spearman/Cosine similarities** with **Wavelet-inspired Conv1D modules**, we achieve near-perfect classification on histopathology benchmarks while staying lightweight enough for edge deployment.

> **Why this implementation?** Unlike vanilla models, we treat **engineering-level safeguards** (dtype-safety, robust ranking, and exponential remapping) as core architectural features to ensure 100% reproducible results.

---

## 📊 Performance at a Glance
Results obtained using our deterministic 70/10/20 stratified splits:

| Dataset | Accuracy | F1-Score | AUC (Macro) |
| :--- | :---: | :---: | :---: |
| **LC25000** (Lung & Colon) | **~100%** | **1.00** | **1.00** |
| **BreakHis** (Breast Cancer) | **97.0%** | **0.96** | **0.98** |
| **NCT-CRC-HE-100K** | **99%+** | **0.99** | **1.00** |

---

## 🏗️ Core Architecture
The model uses a **MobileNetV2** backbone (2.3M parameters)  enhanced by a dual-branch attention strategy:

1. **Statistical Attention Path:** Computes channel-wise similarity (Pearson & Spearman) against an aggregate reference map to identify high-confidence features.
2. **Wavelet-inspired Conv1D:** Treats channel outputs as 1D amplitude sequences to detect local inter-channel motifs.
3. **Calibrated Exponential Remapping:** A grid-tuned mapping function that prevents gradient saturation and ensures stable training.



---

## 🛡️ Stability & Reproducibility (The "Secret Sauce")
This repo isn't just code; it's a reproducible research harness[cite: 492, 653]:
* **Dtype-Safe Ops:** All rank and correlation computations are cast to `float32` with safe-division ($\epsilon=1 \times 10^{-8}$) to avoid NaNs.
* **Robust Spearman Ranking:** Implements batched double-argsort with tie-handling for stable rank estimates.
* **Interpretability:** Built-in **Grad-CAM** pipeline to visualize which tissue regions are driving the model's decisions.

---

## 🛠️ Project Structure
```text
SAWL/
├── sawl_lc2500.ipynb        # 100% Accuracy Pipeline (Lung/Colon)
├── sawl_breakhis_delta.ipynb # High-epoch Stability Testing (Breast Cancer)
├── Presentation.pdf         # Technical breakdown for stakeholders
├── SAWL_Implementation.pdf  # Full methodology & stability analysis
└── README.md                # You are here
