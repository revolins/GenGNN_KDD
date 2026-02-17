import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict
from torch_geometric.utils import dense_to_sparse

from src import utils

# NOTE: This is a large file -- functional pathways between MPNN, GIN, GCN are set with if-checks

class XEyGCNLayer(nn.Module):
    """
    API layer for 
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        dim_ffX: int,
        dim_ffE: int,
        dropout: float = 0.0,
        eps: float = 1e-8,
        use_normalization: bool = True,
        use_dropout: bool = True,
        use_residual: bool = True,
        use_feedforward: bool = True,
        use_node_gating: bool = False,
        use_edge_gating: bool = True,
        ablation_flags: Dict = None,
    ):
        super().__init__()
        self.dx, self.de, self.dy = dx, de, dy
        self.dropout_rate = dropout
        self.eps = eps

        self.use_normalization = use_normalization
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.use_feedforward = use_feedforward
        self.use_node_gating = use_node_gating
        self.use_edge_gating = use_edge_gating

        self.msg = nn.Linear(dx, dx)
        if self.use_edge_gating:
            edge_gate_input_dim = 2 * dx + de + dy
            self.edge_gate_mlp = nn.Sequential(
                nn.Linear(edge_gate_input_dim, max(32, edge_gate_input_dim // 4)),
                nn.GELU(),
                nn.Dropout(dropout if self.use_dropout else 0.0),
                nn.Linear(max(32, edge_gate_input_dim // 4), 1),
            )

        if self.use_node_gating:
            node_gate_input_dim = self.dx + self.de + self.dy
            self.node_gate_mlp = nn.Sequential(
                nn.Linear(node_gate_input_dim, max(32, node_gate_input_dim // 4)),
                nn.GELU(),
                nn.Dropout(dropout if self.use_dropout else 0.0),
                nn.Linear(max(32, node_gate_input_dim // 4), 1),
            )

        self.self_x = nn.Linear(dx, dx)

        if self.use_feedforward:
            self.post_x = nn.Sequential(
                nn.Linear(dx, dim_ffX),
                nn.GELU(),
                nn.Dropout(dropout if self.use_dropout else 0.0),
                nn.Linear(dim_ffX, dx),
                nn.Dropout(dropout if self.use_dropout else 0.0),
            )
        else:
            self.post_x = nn.Identity()

        self.norm_x = nn.LayerNorm(dx) if self.use_normalization else nn.Identity()

        edge_upd_input_dim = 2 * dx + de
        if self.use_feedforward:
            self.edge_upd = nn.Sequential(
                nn.Linear(edge_upd_input_dim, dim_ffE),
                nn.GELU(),
                nn.Dropout(dropout if self.use_dropout else 0.0),
                nn.Linear(dim_ffE, de),
                nn.Dropout(dropout if self.use_dropout else 0.0),
            )
        else:
            self.edge_upd = nn.Sequential(
                nn.Linear(edge_upd_input_dim, de),
                nn.Dropout(dropout if self.use_dropout else 0.0),
            )

        self.norm_e = nn.LayerNorm(de) if self.use_normalization else nn.Identity()

        self.y_upd = nn.Linear(dx + de, dy)
        self.norm_y = nn.LayerNorm(dy) if self.use_normalization else nn.Identity()

        if ablation_flags is None:
            ablation_flags = {}

        # GIN
        if ablation_flags.get("use_gin", False):
            self._gin_enabled = True
            self._gin_learn_eps = ablation_flags.get("gin_learn_eps", True)
            self._gin_edge_mode = ablation_flags.get("gin_edge_mode", "concat")
            self.gin_eps = nn.Parameter(torch.zeros(1), requires_grad=self._gin_learn_eps)
            self.gine_edge_mlp = nn.Sequential(
                nn.Linear(de, max(dx, 32)),
                nn.ReLU(),
                nn.Linear(max(dx, 32), dx),
            )
        else:
            self._gin_enabled = False

        # GCN
        if ablation_flags.get("use_gcn", False):
            self._gcn_enabled = True
            self._gcn_self_loops = ablation_flags.get("gcn_self_loops", True)
            self._gcn_activation = ablation_flags.get("gcn_activation", "gelu")
            self._gcn_use_bias = ablation_flags.get("gcn_use_bias", True)
            self._gcn_use_edge_weights = ablation_flags.get("gcn_use_edge_weights", False)

            self.gcn_lin = nn.Linear(dx, dx, bias=self._gcn_use_bias)
        else:
            self._gcn_enabled = False

    def _act(self, x):
        if self._gcn_activation == "relu":
            return F.relu(x)
        if self._gcn_activation == "gelu":
            return F.gelu(x)
        return x  # "none"

    def forward(self, X, E, y, node_mask):
        """
        X: [B, N, dx]
        E: [B, N, N, de]
        y: [B, dy]
        node_mask: [B, N] or [B, N, 1] (bool or {0,1})
        """
        B, N, _ = X.shape

        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)  # [B,N,1]
        nm = node_mask if node_mask.dtype == torch.bool else (node_mask > 0.5)
        nm_float = nm.float()  # [B,N,1]

        diag_bool = torch.eye(N, device=X.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # [1,N,N,1]
        pair_mask = (nm & nm.transpose(1, 2)).unsqueeze(-1)  # [B,N,N,1]
        pair_mask = pair_mask & (~diag_bool)
        pair_mask_float = pair_mask.float()

        # =================== GIN branch ===================
        if self._gin_enabled:
            weights = pair_mask_float.squeeze(-1)  # [B,N,N]

            if self.use_edge_gating:
                xi = X.unsqueeze(2).expand(-1, -1, N, -1)
                xj = X.unsqueeze(1).expand(-1, N, -1, -1)
                y_b = y.view(B, 1, 1, -1).expand(B, N, N, -1)
                edge_in = torch.cat([xi, xj, E, y_b], dim=-1)

                logits = self.edge_gate_mlp(edge_in)
                logits = logits.masked_fill(~pair_mask, -30.0)
                gates = torch.sigmoid(logits)
                weights = gates.squeeze(-1)  # [B,N,N] - learned attention weights

            if self._gin_edge_mode == "none":
                m = torch.einsum("bij,bjd->bid", weights, X)
            else:
                xj_full = X.unsqueeze(1).expand(-1, N, -1, -1)
                e_ij = self.gine_edge_mlp(E)
                msg_ij = F.relu(xj_full + e_ij) * pair_mask_float
                m = msg_ij.sum(dim=2)

            eps_scalar = (self.gin_eps if self._gin_learn_eps else 0.0)
            agg = (1.0 + eps_scalar) * X + m
            x_new = self.post_x(agg)
            X_out_core = self.norm_x((X + x_new) if self.use_residual else x_new)

            if self.use_node_gating:
                agg_e = (weights.unsqueeze(-1) * E).sum(dim=2) / (weights.sum(dim=2, keepdim=True) + self.eps)
                y_b_node = y.view(B, 1, -1).expand(B, N, -1)
                node_in = torch.cat([X, agg_e, y_b_node], dim=-1)
                node_gates = torch.sigmoid(self.node_gate_mlp(node_in))
                X_out_core = X_out_core * node_gates

            X_out = self.norm_x((X + X_out_core) if self.use_residual else X_out_core)

            xi_out = X_out.unsqueeze(2).expand(-1, -1, N, -1)
            xj_out = X_out.unsqueeze(1).expand(-1, N, -1, -1)
            e_in = torch.cat([xi_out, xj_out, E], dim=-1)
            E_upd = self.edge_upd(e_in)
            E_upd = 0.5 * (E_upd + E_upd.transpose(1, 2))
            E_out = self.norm_e(E + E_upd) if self.use_residual else self.norm_e(E_upd)
            E_out = E_out.masked_fill(diag_bool.expand_as(E_out), 0.0)

            n_valid = nm_float.sum(dim=1).clamp_min(1.0)
            x_pool = (X_out * nm_float).sum(dim=1) / n_valid
            e_valid = pair_mask_float.sum(dim=(1, 2)).clamp_min(1.0)
            e_pool = (E_out * pair_mask_float).sum(dim=(1, 2)) / e_valid
            y_up_input = torch.cat([x_pool, e_pool], dim=-1)
            y_up = self.y_upd(y_up_input)
            Y_out = self.norm_y(y + y_up) if self.use_residual else self.norm_y(y_up)

            X_out = X_out * nm_float
            E_out = E_out * pair_mask_float
            return X_out, E_out, Y_out

        # =================== GCN branch ===================
        if self._gcn_enabled:
            A = pair_mask_float.squeeze(-1)  # [B,N,N]

            if self.use_edge_gating:
                xi = X.unsqueeze(2).expand(-1, -1, N, -1)
                xj = X.unsqueeze(1).expand(-1, N, -1, -1)
                y_b = y.view(B, 1, 1, -1).expand(B, N, N, -1)
                edge_in = torch.cat([xi, xj, E, y_b], dim=-1)

                logits = self.edge_gate_mlp(edge_in)
                logits = logits.masked_fill(~pair_mask, -30.0)
                gates = torch.sigmoid(logits)
                A = gates.squeeze(-1)  # [B,N,N] - learned attention weights
            elif self._gcn_use_edge_weights:
                edge_weights = E[..., 0].masked_fill(~pair_mask.squeeze(-1), 0.0)
                A = A * edge_weights

            if self._gcn_self_loops:
                eye = torch.eye(N, device=X.device).unsqueeze(0)  # [1,N,N]
                valid = (nm.squeeze(-1).float() > 0)              # [B,N]
                A = A + eye * valid.unsqueeze(1) * valid.unsqueeze(2)

            deg = A.sum(-1, keepdim=True).clamp_min(1.0)
            Dm12 = deg.pow(-0.5)

            XW = self.gcn_lin(X)
            XW = XW * Dm12
            H = torch.einsum("bij,bjd->bid", A, XW)
            H = H * Dm12

            if self.use_dropout:
                H = F.dropout(H, p=self.dropout_rate, training=self.training)

            X_out_core = self._act(H)

            if self.use_node_gating:
                agg_e = (A.unsqueeze(-1) * E).sum(dim=2) / (A.sum(dim=2, keepdim=True) + self.eps)
                y_b_node = y.view(B, 1, -1).expand(B, N, -1)
                node_in = torch.cat([X, agg_e, y_b_node], dim=-1)
                node_gates = torch.sigmoid(self.node_gate_mlp(node_in))
                X_out_core = X_out_core * node_gates

            X_out = self.norm_x((X + X_out_core) if self.use_residual else X_out_core)

            xi_out = X_out.unsqueeze(2).expand(-1, -1, N, -1)
            xj_out = X_out.unsqueeze(1).expand(-1, N, -1, -1)
            e_in = torch.cat([xi_out, xj_out, E], dim=-1)

            E_upd = self.edge_upd(e_in)
            E_upd = 0.5 * (E_upd + E_upd.transpose(1, 2))
            E_out = self.norm_e(E + E_upd) if self.use_residual else self.norm_e(E_upd)
            E_out = E_out.masked_fill(diag_bool.expand_as(E_out), 0.0)

            n_valid = nm_float.sum(dim=1).clamp_min(1.0)
            x_pool = (X_out * nm_float).sum(dim=1) / n_valid

            e_valid = pair_mask_float.sum(dim=(1, 2)).clamp_min(1.0)
            e_pool = (E_out * pair_mask_float).sum(dim=(1, 2)) / e_valid

            y_up_input = torch.cat([x_pool, e_pool], dim=-1)
            y_up = self.y_upd(y_up_input)
            Y_out = self.norm_y(y + y_up) if self.use_residual else self.norm_y(y_up)

            X_out = X_out * nm_float
            E_out = E_out * pair_mask_float
            return X_out, E_out, Y_out

        # =================== MPNN/GNN branch ===================
        xi = X.unsqueeze(2).expand(-1, -1, N, -1)
        xj = X.unsqueeze(1).expand(-1, N, -1, -1)
        y_b = y.view(B, 1, 1, -1).expand(B, N, N, -1)
        edge_in = torch.cat([xi, xj, E, y_b], dim=-1)

        if self.use_edge_gating:
            logits = self.edge_gate_mlp(edge_in)
            logits = logits.masked_fill(~pair_mask, -30.0)
            gates = torch.sigmoid(logits)
            weights = gates.squeeze(-1)
        else:
            weights = pair_mask_float.squeeze(-1)

        msg_j = self.msg(X)
        if self.use_dropout:
            msg_j = F.dropout(msg_j, p=self.dropout_rate, training=self.training)

        m = torch.einsum("bij,bjd->bid", weights, msg_j)
        deg = weights.sum(dim=2, keepdim=True)
        m = m / (deg + self.eps)

        if self.use_node_gating:
            agg_e = E.sum(dim=2) / (pair_mask_float.sum(dim=2) + self.eps)  # [B,N,de]
            y_b_node = y.view(B, 1, -1).expand(B, N, -1)
            node_in = torch.cat([X, agg_e, y_b_node], dim=-1)
            node_gates = torch.sigmoid(self.node_gate_mlp(node_in))
            m = m * node_gates

        x_self = self.self_x(X)
        x_core = (X + x_self + m) if self.use_residual else (x_self + m)
        x_core = self.post_x(x_core)
        X_out = self.norm_x((X + x_core) if self.use_residual else x_core)

        xi_out = X_out.unsqueeze(2).expand(-1, -1, N, -1)
        xj_out = X_out.unsqueeze(1).expand(-1, N, -1, -1)
        e_in = torch.cat([xi_out, xj_out, E], dim=-1)

        E_upd = self.edge_upd(e_in)
        E_upd = 0.5 * (E_upd + E_upd.transpose(1, 2))
        E_out = self.norm_e(E + E_upd) if self.use_residual else self.norm_e(E_upd)
        E_out = E_out.masked_fill(diag_bool.expand_as(E_out), 0.0)

        n_valid = nm_float.sum(dim=1).clamp_min(1.0)
        x_pool = (X_out * nm_float).sum(dim=1) / n_valid

        e_valid = pair_mask_float.sum(dim=(1, 2)).clamp_min(1.0)
        e_pool = (E_out * pair_mask_float).sum(dim=(1, 2)) / e_valid

        y_up_input = torch.cat([x_pool, e_pool], dim=-1)
        y_up = self.y_upd(y_up_input)
        Y_out = self.norm_y(y + y_up) if self.use_residual else self.norm_y(y_up)

        X_out = X_out * nm_float
        E_out = E_out * pair_mask_float
        return X_out, E_out, Y_out

class GraphConvolution(nn.Module):
    """
    External API for calling GenGNN layers, reads in ablation flags to alter branches
    """
    def __init__(
        self,
        n_layers: int,
        input_dims: Dict[str, int],
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        output_dims: Dict[str, int],
        act_fn_in: nn.Module,
        act_fn_out: nn.Module,
        dropout: float = 0.0,
        ablation_flags: Dict = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.in_dim_X = input_dims["X"]
        self.in_dim_E = input_dims["E"]
        self.in_dim_y = input_dims["y"]

        self.skip_X = nn.Linear(self.in_dim_X, self.out_dim_X, bias=False)
        self.skip_E = nn.Linear(self.in_dim_E, self.out_dim_E, bias=False)
        self.skip_y = nn.Linear(self.in_dim_y, self.out_dim_y, bias=False)

        if ablation_flags is None:
            ablation_flags = {}

        ablation_flags.setdefault("use_normalization", True)
        ablation_flags.setdefault("use_dropout", True)
        ablation_flags.setdefault("use_residual", True)
        ablation_flags.setdefault("use_feedforward", True)

        ablation_flags.setdefault("use_gin", False)
        ablation_flags.setdefault("gin_learn_eps", True)
        ablation_flags.setdefault("gin_edge_mode", "concat")

        ablation_flags.setdefault("use_edge_gating", True)
        ablation_flags.setdefault("use_node_gating", True)

        ablation_flags.setdefault("use_gcn", False)
        ablation_flags.setdefault("gcn_self_loops", True)
        ablation_flags.setdefault("gcn_activation", "gelu")
        ablation_flags.setdefault("gcn_use_bias", True)
        ablation_flags.setdefault("gcn_use_edge_weights", False)

        self.ablation_flags = ablation_flags

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"] + 64, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.gc_layers = nn.ModuleList(
            [
                XEyGCNLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dropout=dropout,
                    use_normalization=ablation_flags["use_normalization"],
                    use_dropout=ablation_flags["use_dropout"],
                    use_residual=ablation_flags["use_residual"],
                    use_feedforward=ablation_flags["use_feedforward"],
                    use_node_gating=ablation_flags["use_node_gating"],
                    use_edge_gating=ablation_flags["use_edge_gating"],
                    ablation_flags=ablation_flags,
                )
                for _ in range(n_layers)
            ]
        )

        for layer in self.gc_layers:
            layer._gcn_enabled = self.ablation_flags.get("use_gcn", False)
            layer._gcn_self_loops = self.ablation_flags.get("gcn_self_loops", True)
            layer._gcn_activation = self.ablation_flags.get("gcn_activation", "gelu")
            layer._gcn_use_bias = self.ablation_flags.get("gcn_use_bias", True)
            layer._gcn_use_edge_weights = self.ablation_flags.get("gcn_use_edge_weights", False)

            layer._gin_enabled = self.ablation_flags.get("use_gin", False)
            layer._gin_learn_eps = self.ablation_flags.get("gin_learn_eps", True)
            layer._gin_edge_mode = self.ablation_flags.get("gin_edge_mode", "concat")


            if layer._gcn_enabled or layer._gin_enabled:
                layer._gcn_enabled = False
                layer._gin_enabled = False

            if self.ablation_flags.get("use_gcn", False):
                layer._gcn_enabled = True
            elif self.ablation_flags.get("use_gin", False):
                layer._gin_enabled = True

            mode_flags = [
                bool(layer._gcn_enabled),
                bool(layer._gin_enabled)
            ]
            if sum(mode_flags) > 1:
                raise ValueError(
                    "Choose only one backbone per layer."
                )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], self.out_dim_X),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], self.out_dim_E),
        )
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], self.out_dim_y),
        )

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n, device=X.device, dtype=torch.bool)
        diag_mask = ~diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        if self.in_dim_X == self.out_dim_X:
            X_to_out = X[..., : self.out_dim_X]
        else:
            X_to_out = self.skip_X(X)

        if self.in_dim_E == self.out_dim_E:
            E_to_out = E[..., : self.out_dim_E]
        else:
            E_to_out = self.skip_E(E)

        if self.out_dim_y == 0:
            y_to_out = y[..., :0]
        elif self.in_dim_y == self.out_dim_y:
            y_to_out = y[..., : self.out_dim_y]
        else:
            y_to_out = self.skip_y(y)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        time_emb = timestep_embedding(y[:, -1].unsqueeze(-1), 64)
        y_cat = torch.hstack([y, time_emb])

        after_in = utils.PlaceHolder(
            X=self.mlp_in_X(X),
            E=new_E,
            y=self.mlp_in_y(y_cat),
        ).mask(node_mask)

        Xh, Eh, yh = after_in.X, after_in.E, after_in.y
       
        for layer in self.gc_layers:
            Xh, Eh, yh = layer(Xh, Eh, yh, node_mask)

        Eh = 0.5 * (Eh + Eh.transpose(1, 2))

        Xo = self.mlp_out_X(Xh)
        Eo = self.mlp_out_E(Eh)
        yo = self.mlp_out_y(yh)

        Xo = Xo + X_to_out
        Eo = (Eo + E_to_out) * diag_mask
        yo = yo + y_to_out

        Eo = 0.5 * (Eo + Eo.transpose(1, 2))

        return utils.PlaceHolder(X=Xo, E=Eo, y=yo).mask(node_mask)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
