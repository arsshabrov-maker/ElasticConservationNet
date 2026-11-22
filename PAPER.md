# ElasticConservationNet: Amplitude-Driven Edge-Budget Conservation Between Layer Width and Depth

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Standard neural architectures use fixed static topologies where skip connections simply add unnormalized energy across layers. Inspired by mechanical deformation and energy conservation laws, we introduce **ElasticConservationNet**, a PyTorch architecture in which each neuron maintains a constant outgoing connection capacity $\alpha + (1 - \alpha) = 1$, dynamically routing $\sqrt{\alpha}$ of its signal into **layer width** (full projection to layer $L+1$) and $\sqrt{1 - \alpha}$ into **layer depth** (direct downstream propagation along line $i \to i$ across all subsequent layers $L+1 \dots L_{\text{end}}$) based on activation magnitude. By randomly interleaving **50% Type-A neurons** (large activations expand in width; small activations propagate deep) and **50% Type-B neurons** (small activations expand in width; large activations shoot directly forward), the network achieves **95.22% mean accuracy** across 6 benchmarks (3 random seeds, `2,740` avg. params), outperforming both parameter-matched **ResNet** (`94.92%`, `2,928` params) and **MLP** (`94.19%`, `2,928` params), as well as static random edge-conservation ablations (`94.63%`).

## 1. Conservation Law & Dual Polarities
For each unit $i$ in layer $L$ with activation $h_i$ and learnable threshold $\tau_i$:
$$\alpha_A = \sigma\big(2.5(|h_i| - \tau_i)\big), \qquad \alpha_B = 1 - \alpha_A$$
The outgoing signal is split into two orthogonal energy-conserving streams:
$$\mathbf{s}_{\text{width}} = W_L (\mathbf{h} \odot \sqrt{\boldsymbol{\alpha}}), \qquad \mathbf{s}_{\text{depth}}^{(k)} = \mathbf{w}_{L,k} \odot (\mathbf{h} \odot \sqrt{1 - \boldsymbol{\alpha}})$$
At intermediate magnitudes ($\alpha = 0.5$), both width and depth streams operate simultaneously at $\approx 0.707$ amplitude; at extremes, energy shifts smoothly from width to depth or vice versa without inflating signal norm.

## 2. Experimental Verification (6 Benchmarks × 3 Seeds)
* **Symmetric 50/50 Type-A / Type-B Mix (`2,740` params):** **`95.22%`** mean accuracy (`80.40%` on 30%-corrupted Noisy XOR).
* **Standard ResNet (`2,928` params):** `94.92%` mean accuracy (`78.13%` on Noisy XOR).
* **Standard MLP (`2,928` params):** `94.19%` mean accuracy (`74.27%` on Noisy XOR).
* **Control Ablation — Static Random Conservation (`2,740` params):** `94.63%` (confirming that amplitude-driven dynamic routing outperforms static random wiring).
