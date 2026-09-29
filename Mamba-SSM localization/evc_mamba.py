import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from evidential import NIG, nig_uncertainty

try:
    from mamba_ssm import Mamba
    MAMBA_BACKEND = "mamba_ssm"
except ImportError:  # CPU fallback
    from mamba_ref import MambaRef as Mamba
    MAMBA_BACKEND = "reference"
    warnings.warn("mamba_ssm not found - using the pure-PyTorch reference Mamba block.")


class ResidualMambaBlock(nn.Module):
    """Eq. (5): H~ = H + Mamba(H);  H' = LN(H~)  (post-norm residual)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Mamba = in_proj -> depthwise causal conv -> SiLU -> selective SSM, gated by SiLU(z) -> out_proj
        self.mamba = Mamba(d_model=cfg.d_model, d_state=cfg.d_state,
                           d_conv=cfg.d_conv, expand=cfg.expand)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.drop(self.mamba(h)))


class EvidentialHead(nn.Module):
    """phi_f: d -> 4 NIG parameters per velocity axis, constrained by softplus."""

    def __init__(self, d_model: int, num_outputs: int, eps: float = 1e-6):
        super().__init__()
        self.num_outputs = num_outputs
        self.eps = eps
        self.proj = nn.Linear(d_model, 4 * num_outputs)

    def forward(self, h: torch.Tensor) -> NIG:
        g, n, a, b = self.proj(h).view(*h.shape[:-1], self.num_outputs, 4).unbind(-1)
        return NIG(gamma=g,
                   nu=F.softplus(n) + self.eps,             # nu > 0
                   alpha=F.softplus(a) + 1.0 + self.eps,    # alpha > 1 (finite variance)
                   beta=F.softplus(b) + self.eps)           # beta > 0


class EVCMamba(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        # z-score statistics of the OSD channels / targets (set from the training split)
        self.register_buffer("x_mean", torch.zeros(cfg.num_inputs))
        self.register_buffer("x_std", torch.ones(cfg.num_inputs))
        self.register_buffer("y_mean", torch.zeros(cfg.num_outputs))
        self.register_buffer("y_std", torch.ones(cfg.num_outputs))

        self.up_proj = nn.Linear(cfg.num_inputs, cfg.d_model)                       # phi_u
        self.blocks = nn.ModuleList(ResidualMambaBlock(cfg) for _ in range(cfg.n_blocks))
        self.head = EvidentialHead(cfg.d_model, cfg.num_outputs, cfg.eps)            # phi_f

    @torch.no_grad()
    def set_normalization(self, x_mean, x_std, y_mean, y_std):
        for buf, val in ((self.x_mean, x_mean), (self.x_std, x_std),
                         (self.y_mean, y_mean), (self.y_std, y_std)):
            buf.copy_(torch.as_tensor(val, dtype=buf.dtype))
        self.x_std.clamp_(min=1e-6)
        self.y_std.clamp_(min=1e-6)

    def forward(self, osd: torch.Tensor) -> NIG:
        """osd: [B, L, m] raw onboard-sensor window -> NIG in *normalised* target space."""
        h = self.up_proj((osd - self.x_mean) / self.x_std)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1])          # sequence-to-one: last (current) time step

    def normalize_target(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.y_mean) / self.y_std

    @torch.no_grad()
    def predict(self, osd: torch.Tensor):
        """Returns physical-unit (v_hat [B,2], delta [B,2], aleatoric, epistemic).

        Variances are rescaled by y_std^2 back to (m/s)^2.
        """
        return self.denormalize(self(osd))

    def denormalize(self, nig: NIG):
        """NIG (normalised space) -> physical (v_hat, delta, aleatoric, epistemic)."""
        v, var, ale, epi = nig_uncertainty(nig)
        s2 = self.y_std ** 2
        return v * self.y_std + self.y_mean, var * s2, ale * s2, epi * s2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = ModelConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if MAMBA_BACKEND == "mamba_ssm" and dev == "cpu":
        raise SystemExit("mamba_ssm kernels need a CUDA device.")
    net = EVCMamba(cfg).to(dev).eval()
    x = torch.randn(2, cfg.seq_len, cfg.num_inputs, device=dev)
    v, d, a, e = net.predict(x)
    print(f"backend={MAMBA_BACKEND}  params={count_parameters(net)/1e6:.2f}M")
    print("v_hat", v.shape, "delta", d.shape, "all delta > 0:", bool((d > 0).all()))
