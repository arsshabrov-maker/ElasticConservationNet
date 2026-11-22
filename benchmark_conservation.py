import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ElasticConservationNet


class BaselineNet(nn.Module):
  """Parameter-matched MLP and ResNet baselines."""

  def __init__(self, in_dim, out_dim, width=25, depth=4, mode='mlp'):
    super().__init__()
    self.width = width
    self.depth = depth
    self.mode = mode
    self.in_proj = nn.Linear(in_dim, width)
    self.layers = nn.ModuleList(
        [nn.Linear(width, width) for _ in range(depth - 1)]
    )
    self.head = nn.Linear(width, out_dim)

  def forward(self, x):
    h = F.gelu(self.in_proj(x))
    for layer in self.layers:
      out = layer(h)
      h = F.gelu(out + h if self.mode == 'resnet' else out)
    return self.head(h)


def build_6_benchmarks():
  raw = []
  d = load_digits()
  raw.append(('1. Digits 8x8', d.data, d.target, 10))
  d = load_breast_cancer()
  raw.append(('2. Breast Cancer', d.data, d.target, 2))
  d = load_wine()
  raw.append(('3. Wine Chem', d.data, d.target, 3))
  X, y = make_circles(n_samples=800, noise=0.12, factor=0.45, random_state=42)
  raw.append(('4. Nested Circles', X, y, 2))

  rng = np.random.RandomState(42)
  X5 = rng.randn(1500, 16).astype(np.float32)
  cols = [X5[:, i] for i in range(16)]
  y5 = (
      np.tanh(cols.pop(0) + cols.pop(0) - cols.pop(0) + cols.pop(0)) * 2.0
      + np.sin(0.5 * cols.pop(0) * cols.pop(0))
      + np.cos(0.5 * cols.pop(0) - cols.pop(0))
      > 0.2
  ).astype(np.int64)
  raw.append(('5. Depth + Width', X5, y5, 2))

  X6 = rng.randn(1000, 10).astype(np.float32)
  c0, c1 = X6[:, 0], X6[:, 1]
  y6 = ((c0 * c1) > 0).astype(np.int64)
  mask = rng.rand(1000) < 0.30
  X6[mask] += rng.randn(mask.sum(), 10).astype(np.float32) * 5.0
  raw.append(('6. Noisy XOR 30%', X6, y6, 2))

  prepared = []
  for name, X_r, y_r, n_o in raw:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_r, y_r, test_size=0.25, random_state=42
    )
    sc = StandardScaler()
    prepared.append((
        name,
        torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(sc.transform(X_te), dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long),
        n_o,
    ))
  return prepared


def main():
  tasks = build_6_benchmarks()
  print('=' * 84)
  print(
      'ELASTIC CONSERVATION NET (50/50 A/B) vs PARAMETER-MATCHED RESNET & MLP'
  )
  print('=' * 84)

  totals = {'Elastic 50/50': [], 'ResNet': [], 'MLP': []}
  for name, X_tr, y_tr, X_te, y_te, n_o in tasks:
    in_dim = X_tr.size(1)
    row_res = {}
    for label, builder in [
        ('Elastic 50/50', lambda sd: ElasticConservationNet(in_dim, n_o, 24, 4, sd)),
        ('ResNet', lambda sd: BaselineNet(in_dim, n_o, 25, 4, 'resnet')),
        ('MLP', lambda sd: BaselineNet(in_dim, n_o, 25, 4, 'mlp')),
    ]:
      accs = []
      for sd in (42, 100, 777):
        torch.manual_seed(sd)
        net = builder(sd)
        opt = torch.optim.AdamW(net.parameters(), lr=0.008, weight_decay=1e-3)
        best = 0.0
        for _ in range(160):
          net.train()
          opt.zero_grad()
          F.cross_entropy(net(X_tr), y_tr).backward()
          opt.step()
          net.eval()
          with torch.no_grad():
            best = max(
                best,
                (net(X_te).argmax(dim=1) == y_te).float().mean().item() * 100.0,
            )
        accs.append(best)
      mean_a = float(np.mean(accs))
      row_res[label] = mean_a
      totals[label].append(mean_a)

    print(
        f'{name:<20} | Elastic 50/50: {row_res["Elastic 50/50"]:6.2f}% |'
        f' ResNet: {row_res["ResNet"]:6.2f}% | MLP: {row_res["MLP"]:6.2f}%'
    )

  print('=' * 84)
  print(
      f'MEAN (6 TASKS x 3 SEEDS) | Elastic 50/50:'
      f' {np.mean(totals["Elastic 50/50"]):6.2f}% | ResNet:'
      f' {np.mean(totals["ResNet"]):6.2f}% | MLP:'
      f' {np.mean(totals["MLP"]):6.2f}%'
  )
  print('=' * 84)


if __name__ == '__main__':
  main()
