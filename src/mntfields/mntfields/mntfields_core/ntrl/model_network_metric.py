import math
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Linear, InstanceNorm1d
from torch import Tensor
from torch.autograd import grad

torch.backends.cudnn.benchmark = True


def sigmoid_out(input: Tensor) -> Tensor:
    """
    Output squashing function for final scalar prediction.
    Uses a low-slope sigmoid to keep values smooth and bounded.
    """
    return torch.sigmoid(0.1 * input)


class Sigmoid_out(torch.nn.Module):
    """
    Module wrapper for sigmoid_out, used for clarity and consistency.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid_out(input)


class NN(torch.nn.Module):
    """
    Neural metric network for mNTFields.

    Learns a smooth travel-time / metric function τ(x0, x1)
    using Fourier features, gated sinusoidal layers, and
    Lipschitz-controlled weights.

    Input:
        coords: Tensor of shape (N, 2*dim)
                concatenated start and end points

    Output:
        tau:    Tensor of shape (N, 1)
                predicted travel time / metric value
    """

    def __init__(self, device: str, dim: int, B: Tensor):
        super(NN, self).__init__()

        self.dim = dim
        # Hidden layer size
        h_size = 256

        # Fourier feature matrix (transposed for right-multiplication)
        self.B = B.T.to(device)
        # Pre-compute scaled B — avoids a multiplication every forward call
        self._fourier_w = (2.0 * math.pi * self.B)

        # Activation and output scaling
        self.scale = 10
        self.act = torch.nn.Softplus(beta=self.scale)
        self.actout = Sigmoid_out()

        # Number of gated residual blocks
        self.nl1 = 2

        # Encoder layers
        self.encoder = torch.nn.ModuleList()
        self.encoder_norm = InstanceNorm1d(h_size)

        # Input layer: Fourier features → hidden
        #self.encoder.append(Linear(252, h_size))
        fourier_dim = 2 * self.B.shape[1]  # B is (128, 3), so fourier_dim = 256
        self.encoder.append(Linear(fourier_dim, h_size))
        # Gated sinusoidal blocks
        for _ in range(3 * self.nl1):
            self.encoder.append(Linear(h_size, h_size))

        # Final linear projection before latent comparison
        self.encoder.append(Linear(h_size, h_size))

        # Learnable gates controlling residual blending
        self.gate = torch.nn.ModuleList()
        for _ in range(self.nl1):
            self.gate.append(Linear(1, 1))

        # Positional-encoding gates
        self.pe_gate = torch.nn.ModuleList()
        self.pe_gate.append(Linear(h_size, h_size))
        self.pe_gate.append(Linear(h_size, h_size))

    def init_weights(self, m):
        """
        Truncated normal initialization for linear layers.
        Gates are initialized to zero to start from identity behavior.
        """
        if isinstance(m, torch.nn.Linear):
            stdv = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1)))
            torch.nn.init.trunc_normal_(
                m.weight,
                mean=0.0,
                std=stdv,
                a=-2.0 * stdv,
                b=2.0 * stdv
            )
            m.bias.data.fill_(0)

        for i in range(self.nl1):
            self.gate[i].weight.data.fill_(0)
            self.gate[i].bias.data.fill_(0)

    def input_mapping(self, x: Tensor) -> Tensor:
        """
        Fourier feature mapping for high-frequency spatial encoding.
        """
        x_proj = x @ self._fourier_w
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def lip_norm(self, w: Tensor) -> Tensor:
        """
        Soft Lipschitz normalization for stabilizing gradients.
        Prevents exploding derivatives during online training.
        """
        absrowsum = torch.sqrt(torch.sum(w ** 2, dim=1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1)

    def out(self, coords: Tensor):
        """
        Core forward pass computing τ(x0, x1).

        Returns:
            tau   : predicted travel time (N, 1)
            w     : last-layer weights (used in loss)
            coords: input with gradient tracking enabled
        """
        #coords = coords.clone().detach().requires_grad_(True)
        coords = coords.clone().detach().requires_grad_(True)

        size = coords.shape[0]

        # Split start and end points
        x0 = coords[:, :self.dim]
        x1 = coords[:, self.dim:]

        # Stack for shared encoding
        x = torch.vstack((x0, x1))
        x = self.input_mapping(x)

        # Positional encoding gates
        w = self.lip_norm(self.pe_gate[0].weight)
        b = self.pe_gate[0].bias
        u = torch.sin(x @ w.T + b)

        w = self.lip_norm(self.pe_gate[1].weight)
        b = self.pe_gate[1].bias
        v = torch.sin(x @ w.T + b)

        # Gated sinusoidal blocks
        for ii in range(self.nl1):
            x_tmp = x

            w = self.lip_norm(self.encoder[3 * ii + 1].weight)
            b = self.encoder[3 * ii + 1].bias
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = self.lip_norm(self.encoder[3 * ii + 2].weight)
            b = self.encoder[3 * ii + 2].bias
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = self.lip_norm(self.encoder[3 * ii + 3].weight)
            b = self.encoder[3 * ii + 3].bias
            y = x @ w.T + b

            weight = torch.sigmoid(0.1 * self.gate[ii].weight)
            x = (1 - weight) * x_tmp + weight * torch.sin(y)

        # Final projection
        w = self.lip_norm(self.encoder[-1].weight)
        b = self.encoder[-1].bias
        y = x @ w.T + b
        y = self.encoder_norm(y)

        # Separate start/end embeddings
        x0 = y[:size, ...]
        x1 = y[size:, ...]

        # Metric construction via soft minimum pooling
        x = torch.sqrt((x0 - x1) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, dim=2) - np.log(16)) / 10
        x = 0.2 * torch.sum(x, dim=1, keepdim=True)
        
        #x = torch.clamp(x, min=-10.0, max=10.0)  # Before sigmoid - clamping
        #x = torch.sigmoid(x)  # Maps to (0, 1)
        #x = torch.clamp(x, min=1e-4, max=0.99)  # After sigmoid

        return x, w, coords

    def forward(self, coords: Tensor):
        """
        Standard PyTorch forward wrapper.
        """
        coords = coords.clone().detach().requires_grad_(True)
        output, w, coords = self.out(coords)
        return output, coords
