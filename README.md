# ElasticConservationNet: Amplitude-Driven Edge-Budget Conservation Between Width and Depth

**ElasticConservationNet** replaces static neural wiring with an **elastic conservation law** ($\alpha + (1 - \alpha) = 1$). Depending on signal magnitude $|h|$, each neuron dynamically shifts its outgoing capacity between **layer width** ($\sqrt{\alpha}$ into the next dense layer) and **layer depth** ($\sqrt{1 - \alpha}$ along a direct $i \to i$ line across all downstream layers).

---

## 💡 Core Mechanism: Symmetric Type-A / Type-B Polarities

Each hidden layer randomly interleaves two complementary neuron polarities (50% / 50%) under a strict energy-conservation constraint ($g_{\text{width}}^2 + g_{\text{depth}}^2 = 1$):

* **Type A (`Large → Width, Small → Depth`):** High-magnitude features fan out across the entire next layer for broad mixing, while subtle low-magnitude signals travel undisturbed down a direct line to deep layers.
* **Type B (`Small → Width, Large → Depth`):** Weak signals scatter across width to gather context, while strong confident features shoot directly toward the output without being diluted by neighbors.
* **Mid-Range Balance ($\alpha \approx 0.5$):** Both width and depth streams open simultaneously at $\approx 71\%$ capacity.

---

## 📊 Experimental Results (6 Benchmarks × 3 Random Seeds)

Evaluated across *Digits 8×8, Breast Cancer, Wine Chemistry, Nested Circles, Mixed Depth+Width, and Noisy XOR (30% Extreme Outliers)*:

| Architecture | Avg. Params | Mean 6-Task Acc (3 Seeds) ↑ | Noisy XOR (30% Outliers) ↑ |
| :--- | :---: | :---: | :---: |
| **ElasticConservationNet (50% A / 50% B)** | **2,740** | **95.22%** | **80.40%** |
| Elastic Mix (80% A / 20% B) | 2,740 | 94.98% | 79.33% |
| Classical ResNet (Unnormalized Skip) | 2,928 | 94.92% | 78.13% |
| Classical MLP (Width-Only Baseline) | 2,928 | 94.19% | 74.27% |

### Control Ablation: Dynamic Amplitude Routing vs. Static Random Wiring
To verify that gains come from **amplitude-driven dynamic conservation** rather than mere sparse skip connectivity:

| Routing Mechanism (Equal 2,740 Params) | Mean 6-Task Acc ↑ | Noisy XOR (30% Outliers) ↑ |
| :--- | :---: | :---: |
| Static Random Conservation (fixed random $\alpha$) | 94.63% | 77.07% |
| Uniform 50/50 Conservation (constant $\alpha = 0.5$) | 94.75% | 76.67% |
| **Dynamic 50/50 Type-A/B Conservation (Ours)** | **94.96%** | **78.00%** |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python benchmark_conservation.py
```

