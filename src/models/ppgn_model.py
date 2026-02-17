import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch import Tensor

from src import utils

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _pair_mask(node_mask: Tensor) -> Tensor:
    # node_mask: [B,n] in {0,1}
    x = node_mask.unsqueeze(-1)            # [B,n,1]
    return (x.unsqueeze(2) * x.unsqueeze(1))  # [B,n,n,1]

def _masked_mean(t: Tensor, mask: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den

def _symmetrize_pair(H: Tensor) -> Tensor:
    return 0.5 * (H + H.transpose(1, 2))


class MLP(nn.Module):
    """Simple MLP for last-dim features."""
    def __init__(self, d_in: int, hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LocalPPGNLayer(nn.Module):
    """
      Local PPGN layer:
      m = (mlp1(H)*mask) @ (mlp2(H)*mask) / sqrt(|V|)
      H = mlp3([H, m]) * mask
      E = E + edge_upd(H) with symmetrization
    """

    def __init__(self, h: int, de: int, p: int, mlp_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.mlp1 = MLP(h, mlp_hidden, p, dropout)
        self.mlp2 = MLP(h, mlp_hidden, p, dropout)
        self.mlp3 = MLP(h + p, mlp_hidden, h, dropout)
        self.edge_upd = nn.Linear(h, de)

    def forward(self, H: Tensor, E: Tensor, mask: Tensor, node_mask: Tensor):
        """
        H:    [B,n,n,h]
        E:    [B,n,n,de]
        mask: [B,n,n,1]
        node_mask: [B,n]
        """
        m1 = (self.mlp1(H) * mask).permute(0, 3, 1, 2)  # [B,p,n,n]
        m2 = (self.mlp2(H) * mask).permute(0, 3, 1, 2)  # [B,p,n,n]
        m = m1 @ m2                                      # [B,p,n,n]

        size = node_mask.sum(-1).clamp_min(1).float()     # [B]
        m = m / size.sqrt().view(-1, 1, 1, 1)

        m = m.permute(0, 2, 3, 1)                         # [B,n,n,p]
        H = self.mlp3(torch.cat([H, m], dim=-1)) * mask   # [B,n,n,h]
        H = _symmetrize_pair(H) * mask

        E_upd = self.edge_upd(H) * mask
        E_upd = 0.5 * (E_upd + E_upd.transpose(1, 2))
        E = E + E_upd
        E = _symmetrize_pair(E) * mask

        return H, E


class GraphPPGN(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: nn.ReLU(),
        act_fn_out: nn.ReLU(),
        dropout: float = 0.0,
        gen_model: str = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        dx = hidden_dims["dx"]
        de = hidden_dims["de"]
        dy = hidden_dims["dy"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], dx),
            act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], de),
            act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"] + 64, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], dy),
            act_fn_in,
        )

        h = hidden_dims.get("d_pair", max(dx, de))
        p = hidden_dims.get("ppgn_features", h)
        mlp_hidden = hidden_dims.get("dim_ff_pair", 256)
        dropout = dropout

        self.h = h
        self.h_skip = (n_layers + 1) * h

        self.pair_lift = MLP(
            d_in=(2 * dx + de + dy),
            hidden=mlp_hidden,
            d_out=h,
            dropout=dropout,
        )

        self.ppgn_layers = nn.ModuleList(
            [LocalPPGNLayer(h=h, de=de, p=p, mlp_hidden=mlp_hidden, dropout=dropout) for _ in range(n_layers)]
        )

        self.node_out = nn.Sequential(
            nn.Linear(dx + 2 * self.h_skip + dy, hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(de, hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.y_out = nn.Sequential(
            nn.Linear(dy + h + dx, hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

        # Residual projection layers if input and output dims differ
        if input_dims["X"] != output_dims["X"] and "vgae" in gen_model:
            self.res_proj_X = nn.Linear(input_dims["X"], output_dims["X"])
        if input_dims["E"] != output_dims["E"] and "vgae" in gen_model:
            self.res_proj_E = nn.Linear(input_dims["E"], output_dims["E"])
        if input_dims["y"] != output_dims["y"] and "vgae" in gen_model:
            self.res_proj_y = nn.Linear(input_dims["y"], output_dims["y"])
        # h-skip readout in case of stability issues from full y
        # self.y_out = nn.Sequential(
        #     nn.Linear(dy + self.h_skip + dx, hidden_mlp_dims["y"]),
        #     act_fn_out,
        #     nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        # )

    def forward(self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor):
        import src.utils as utils

        B, n = X.shape[0], X.shape[1]
        pmask = _pair_mask(node_mask)          # [B,n,n,1]
        xmask = node_mask.unsqueeze(-1)        # [B,n,1]

        diag_mask = torch.eye(n, device=E.device).bool()
        diag_mask = (~diag_mask).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        if hasattr(self, 'res_proj_X'):
            X_to_out = self.res_proj_X(X_to_out)
        if hasattr(self, 'res_proj_E'):
            E_to_out = self.res_proj_E(E_to_out)
        if hasattr(self, 'res_proj_y'):
            y_to_out = self.res_proj_y(y_to_out)

        E_enc = self.mlp_in_E(E)
        E_enc = 0.5 * (E_enc + E_enc.transpose(1, 2))
        E_enc = E_enc * pmask

        time_emb = timestep_embedding(y[:, -1].unsqueeze(-1), 64)
        y_aug = torch.hstack([y, time_emb])
        y_enc = self.mlp_in_y(y_aug)  # [B,dy]

        X_enc = self.mlp_in_X(X) * xmask

        Xi = X_enc.unsqueeze(2).expand(B, n, n, X_enc.size(-1))
        Xj = X_enc.unsqueeze(1).expand(B, n, n, X_enc.size(-1))
        yij = y_enc.unsqueeze(1).unsqueeze(1).expand(B, n, n, y_enc.size(-1))

        H = self.pair_lift(torch.cat([Xi, Xj, E_enc, yij], dim=-1)) * pmask
        H = _symmetrize_pair(H) * pmask

        skips = [H]
        for layer in self.ppgn_layers:
            H, E_enc = layer(H, E_enc, pmask, node_mask)
            skips.append(H)

        H_skip = torch.cat(skips, dim=-1)      # [B,n,n,(L+1)*h]
        H_skip = _symmetrize_pair(H_skip) * pmask

        E_out = self.mlp_out_E(E_enc) * pmask
        E_out = 0.5 * (E_out + E_out.transpose(1, 2))
        E_out = E_out * diag_mask

        row_mean = _masked_mean(H_skip, pmask, dim=2)  # [B,n,h_skip]
        col_mean = _masked_mean(H_skip, pmask, dim=1)  # [B,n,h_skip]
        yx = y_enc.unsqueeze(1).expand(B, n, y_enc.size(-1))

        X_out = self.node_out(torch.cat([X_enc, row_mean, col_mean, yx], dim=-1)) * xmask

        H_pool_n = _masked_mean(H, pmask, dim=1)                 # [B,n,h]
        H_pool = _masked_mean(H_pool_n, node_mask.unsqueeze(-1), dim=1)  # [B,h]
        X_pool = _masked_mean(X_enc, xmask, dim=1)               # [B,dx]
        y_out = self.y_out(torch.cat([y_enc, H_pool, X_pool], dim=-1))   # [B,out_dy]

        X_out = X_out + X_to_out
        E_out = 0.5 * (E_out + E_out.transpose(1, 2))
        E_out = (E_out + E_to_out) * diag_mask
        y_out = y_out + y_to_out
       
        E_out = 0.5 * (E_out + E_out.transpose(1, 2))
        return utils.PlaceHolder(X=X_out, E=E_out, y=y_out).mask(node_mask)
