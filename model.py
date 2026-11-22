import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElasticConservationNet(nn.Module):
  """
    Elastic Edge-Budget Conservation Network (`ElasticConservationNet`):
      Each neuron preserves a constant outgoing connection energy budget:
        alpha + (1 - alpha) = 1
      routing sqrt(alpha) into layer WIDTH (next layer) and sqrt(1 - alpha)
      into layer DEPTH (direct line i -> i across all downstream layers).

      Symmetric 50% Type-A / 50% Type-B Polarity:
        - Type A: Large activations -> Width; Small activations -> Depth line
        - Type B: Small activations -> Width; Large activations -> Depth line
    """

  def __init__(self, in_dim, out_dim, width=24, depth=4, seed=42):
    super().__init__()
    self.width = width
    self.depth = depth

    self.in_proj = nn.Linear(in_dim, width)
    self.width_layers = nn.ModuleList(
        [nn.Linear(width, width) for _ in range(depth - 1)]
    )
    self.depth_line_weights = nn.ParameterList([
        nn.Parameter(torch.randn(depth - 1 - l, width) * 0.35 + 0.5)
        for l in range(depth - 1)
    ])
    self.balance_thresholds = nn.ParameterList(
        [nn.Parameter(torch.ones(1, width) * 0.6) for _ in range(depth - 1)]
    )

    rng = np.random.RandomState(seed)
    fixed_polarities = []
    for _ in range(depth - 1):
      n_a = width // 2
      n_b = width - n_a
      arr = np.concatenate(
          [np.ones(n_a, dtype=np.float32), -np.ones(n_b, dtype=np.float32)]
      )
      rng.shuffle(arr)
      fixed_polarities.append(arr)

    self.register_buffer(
        'fixed_pol',
        torch.tensor(np.array(fixed_polarities), dtype=torch.float32),
    )
    self.head = nn.Linear(width, out_dim)

  def forward(self, x):
    B = x.size(0)
    layer_inputs = [self.in_proj(x)] + [
        torch.zeros(B, self.width, device=x.device)
        for _ in range(self.depth - 1)
    ]

    for l in range(self.depth - 1):
      h = F.gelu(layer_inputs[l])
      mag = torch.abs(h)
      thresh = F.softplus(self.balance_thresholds[l])

      # Type A: large -> width; Type B: small -> width
      alpha_a = torch.sigmoid(2.5 * (mag - thresh))
      alpha_b = 1.0 - alpha_a

      p = self.fixed_pol[l].unsqueeze(0)
      w_mix = 0.5 * (p + 1.0)
      alpha = w_mix * alpha_a + (1.0 - w_mix) * alpha_b

      # Energy conservation gates: w_gate^2 + d_gate^2 = 1
      w_gate = torch.sqrt(alpha + 1e-6)
      d_gate = torch.sqrt(1.0 - alpha + 1e-6)

      # 1. Width stream (to all units in layer l + 1)
      layer_inputs[l + 1] = layer_inputs[l + 1] + self.width_layers[l](
          h * w_gate
      )

      # 2. Depth-line stream (along i -> i across all downstream layers)
      long_sig = h * d_gate
      line_w = self.depth_line_weights[l]
      for step in range(self.depth - 1 - l):
        target_l = l + 1 + step
        layer_inputs[target_l] = (
            layer_inputs[target_l] + long_sig * line_w[step].unsqueeze(0)
        )

    return self.head(F.gelu(layer_inputs[-1]))
