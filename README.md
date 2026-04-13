# Hydra: Reproduction and Improvement Study

**COMP41850 AI for Time Series — Group Project**

Reproduction and improvement of:
> Dempster, A., Schmidt, D.F., Webb, G.I. (2022). *Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification.* arXiv:2203.13652
> [[Paper]](https://arxiv.org/abs/2203.13652) [[Original Code]](https://github.com/angus924/hydra)

---

## Repository Structure

```
hydra-reproduction/
├── notebook.ipynb           # Main project notebook (all experiments)
├── baseline_results.csv     # Baseline reproduction results
├── improvement_results.csv  # Improvement experiment results
├── sensitivity_kg.csv       # Hyperparameter sensitivity results
├── report.pdf               # Final IEEE-format report
└── README.md
```

---

## What the Notebook Does

The notebook is divided into 13 sections:

| Section | Description |
|---------|-------------|
| 1 | Environment setup and dependency installation |
| 2 | Core Hydra implementation (faithful reproduction from paper) |
| 3 | Data loading from UCR archive via `aeon` |
| 4 | Baseline reproduction experiments (k=8, g=64) |
| 5 | Hyperparameter sensitivity analysis (k and g sweep) |
| 6 | Proposed improvements (4 variants) |
| 7 | Evaluation and results comparison |
| 8 | Visualisations (accuracy plots, sensitivity heatmaps) |
| 9–11 | Extended evaluation across UCR datasets |
| 12 | Statistical testing (Wilcoxon signed-rank) |
| 13 | Summary and conclusions |

---

## How to Reproduce

### Option A — Kaggle (Recommended, free GPU)

This notebook was developed and tested on Kaggle with a Tesla T4 GPU.

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
2. Click **Create** → **New Notebook**
3. Upload `notebook.ipynb` via File → Import Notebook
4. Under **Settings** (right panel) → **Accelerator** → select **GPU T4 x2**
5. Enable internet access under **Settings** → **Internet**
6. Click **Run All**

Expected runtime: ~30–60 minutes with GPU.

### Option B — Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `notebook.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Runtime → Run all

### Option C — Local (CPU only)

```bash
# Install dependencies
pip install torch scikit-learn numpy pandas matplotlib seaborn aeon tslearn sktime scipy

# Launch notebook
jupyter notebook notebook.ipynb
```

> **Note:** Running locally on CPU will be significantly slower (especially for larger datasets). The notebook is optimised for GPU but will fall back to CPU automatically.

---

## Improvements Implemented

| Improvement | Description | Result vs Baseline |
|-------------|-------------|-------------------|
| Ensemble Hydra (5 members) | Average predictions across 5 random seeds | More stable, marginal gain |
| Second-Order Difference | Adds acceleration (diff of diff) to feature set | Best accuracy overall |
| L1 Feature Selection | Logistic regression with L1 penalty for sparse weighting | Statistically significant improvement |
| Multi-Scale Hydra | Combines multiple (k, g) configurations | Broader pattern coverage |

Statistical significance tested using the Wilcoxon signed-rank test (as used in the original paper).

---

## Key Results

**Baseline Hydra (k=8, g=64):**
- Mean accuracy across 10 datasets: **0.9200**
- Confirms paper's reported best k/g configuration

**Best improvement:** Second-Order Difference
- Mean accuracy: **0.9371**
- Adds acceleration information complementary to first-order difference

**Reproduction findings confirmed:**
- Default k=8, g=64 achieves the best accuracy/efficiency trade-off ✓
- First-order difference consistently improves accuracy ✓
- Counting both max+min responses outperforms max-only ✓
- Combined soft+hard counting is best ✓
- Clipping has minimal effect at optimal k/g ✓

---

## Requirements

```
torch >= 2.0
scikit-learn >= 1.0
numpy >= 1.24
pandas >= 1.5
matplotlib
seaborn
aeon >= 0.9
scipy
```

GPU is recommended but not required. The notebook auto-detects CUDA availability and falls back to CPU.

---

## Citation

```bibtex
@article{dempster2022hydra,
  title={Hydra: Competing convolutional kernels for fast and accurate time series classification},
  author={Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  journal={arXiv preprint arXiv:2203.13652},
  year={2022}
}
```
