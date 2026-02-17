import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from .gnn_model import GraphConvolution


class GenGNNFeatureExtractor(nn.Module):
    """
    A dedicated feature extractor based on GenGNN for graph representation learning.

    This model is currently used as a randomly-initialized feature-extraction
    tool in downstream tasks like MagDiff evaluation.
    """

    def __init__(self, cfg, dataset_infos=None):
        """
        Initialize the GenGNN Feature Extractor.

        Args:
            cfg: Configuration object with feature_extractor parameters
            dataset_infos: Dataset information with input/output dimensions
        """
        super().__init__()

        fe_cfg = cfg.feature_extractor
        self.n_layers = fe_cfg.n_layers
        self.num_mlp_layers = getattr(fe_cfg, 'num_mlp_layers', 2) # GIN MLP
        self.hidden_dims = fe_cfg.hidden_dims
        self.dropout = fe_cfg.dropout
        self.final_dropout = getattr(fe_cfg, 'final_dropout', 0.0)
        self.num_classes = fe_cfg.num_classes
        self.graph_pooling_type = getattr(fe_cfg, 'graph_pooling_type', 'mean')
        self.neighbor_pooling_type = getattr(fe_cfg, 'neighbor_pooling_type', 'sum')
        self.learn_eps = getattr(fe_cfg, 'learn_eps', False)
        self.edge_feat_dim = getattr(fe_cfg, 'edge_feat_dim', 0)
        self.orthogonal_init = getattr(fe_cfg, 'orthogonal_init', True)

        if dataset_infos is not None:
            self.input_dim = dataset_infos.input_dims['X']
            self.edge_dim = dataset_infos.input_dims['E']
            self.y_dim = dataset_infos.input_dims['y']
        else:
            self.input_dim = getattr(fe_cfg, 'input_dim', self.hidden_dims)
            self.edge_dim = getattr(fe_cfg, 'edge_dim', 0)
            self.y_dim = 0

        self.edge_dim_eff = self.edge_dim
        self.y_dim = max(self.y_dim, 1)
        self.y_dim_eff = self.y_dim

        # GenGNN pipeline ablation flags -- set to GIN
        ablation_flags = {
            "use_gin": True,
            "gin_learn_eps": self.learn_eps,
            "gin_edge_mode": "concat" if self.edge_feat_dim > 0 else "none",
            "use_normalization": True,
            "use_dropout": True,
            "use_residual": True,
            "use_feedforward": True,
            "use_node_gating": False,  # Disable gating for pure GIN
            "use_edge_gating": False,
            "use_gcn": False
        }

        self.encoders = nn.ModuleList()
        for i in range(self.n_layers):
            encoder = GraphConvolution(
                n_layers=i + 1, 
                input_dims={
                    'X': self.input_dim,
                    'E': self.edge_dim_eff,
                    'y': self.y_dim_eff
                },
                hidden_mlp_dims={
                    'X': self.hidden_dims,
                    'E': self.hidden_dims,
                    'y': self.hidden_dims 
                },
                hidden_dims={
                    'dx': self.hidden_dims,
                    'de': self.hidden_dims,
                    'dy': self.hidden_dims,
                    'n_head': 4,  # Arbitrary values for compatibility
                    'dim_ffX': self.hidden_dims,
                    'dim_ffE': self.hidden_dims
                },
                output_dims={
                    'X': self.hidden_dims,
                    'E': self.edge_dim_eff,
                    'y': self.y_dim_eff
                },
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
                dropout=self.dropout,
                ablation_flags=ablation_flags
            )
            self.encoders.append(encoder)

        self.encoder = self.encoders[-1]

        if self.graph_pooling_type == 'sum':
            self.pooling = global_add_pool
        elif self.graph_pooling_type == 'mean':
            self.pooling = global_mean_pool
        elif self.graph_pooling_type == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(f"Unknown graph_pooling_type: {self.graph_pooling_type}")

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(self.input_dim, self.num_classes))
            else:
                self.linears_prediction.append(
                    nn.Linear(self.hidden_dims, self.num_classes))

        self.drop = nn.Dropout(self.final_dropout)

        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, self.input_dim * self.edge_dim)
        )

        if self.orthogonal_init:
            self.apply_orthogonal_initialization()

    def forward(self, noisy_data, extra_data=None, node_mask=None):
        """
        Forward pass for diffusion-style training (predicting clean graphs from noisy ones).

        Args:
            noisy_data: Dictionary with noisy graph data (same format as diffusion model)
            extra_data: Extra features (same as diffusion model)
            node_mask: Node mask tensor

        Returns:
            predictions: Dictionary with predicted X, E, y (same as diffusion model)
        """

        if hasattr(noisy_data, 'x'): 
            return self.forward_pyg(noisy_data)

        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()

        pred = self.encoder(X, E, y, node_mask)

        return pred

    def forward_pyg(self, data):
        """
        Forward pass for PyG Data objects (for feature extraction).
        Matches the GIN architecture with layer-wise predictions.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch

        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        hidden_rep = [data.x]

        for encoder in self.encoders:
            node_embeds = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            hidden_rep.append(node_embeds.X)

        score_over_layer = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pooling(h, data.batch)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer

    def extract_features(self, data):
        """
        Extract graph features without classification head.
        Returns concatenated layer representations (matching GIN's get_graph_embed).

        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch

        Returns:
            features: Concatenated graph embeddings from all layers [batch_size, (n_layers+1) * hidden_dims]
        """
        from src import utils

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        X, E = dense_data.X, dense_data.E

        y = torch.zeros(X.size(0), self.y_dim_eff).to(X.device)  # Dummy y for compatibility

        hidden_rep = [X]

        for encoder in self.encoders:
            node_embeds = encoder(X, E, y, node_mask)
            hidden_rep.append(node_embeds.X)

        graph_embed = torch.Tensor([]).to(X.device)
        for i, h in enumerate(hidden_rep):
            h_masked = h * node_mask.unsqueeze(-1)
            h_flat = h_masked.view(-1, h.size(-1))
            valid_mask = node_mask.view(-1).bool()
            if valid_mask.numel() == 0 or valid_mask.sum() == 0:
                continue
            h_valid = h_flat[valid_mask]
            batch = torch.arange(X.size(0), device=X.device).repeat_interleave(
                torch.sum(node_mask, dim=1).long()
            )
            if batch.numel() != h_valid.size(0):
                # fall back to using boolean mask to match sizes
                batch = torch.arange(X.size(0), device=X.device).repeat_interleave(
                    torch.sum(node_mask, dim=1).long(),
                    dtype=torch.long
                )
                batch = batch[: h_valid.size(0)]

            pooled_h = self.pooling(h_valid, batch)
            graph_embed = torch.cat([graph_embed, pooled_h], dim=1)

        return graph_embed
    

    def reconstruct_graph(self, data):
        """
        Self-supervised reconstruction task.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch

        Returns:
            reconstructed_edges: Predicted edge features [batch_size, max_nodes, max_nodes, edge_dim]
        """
        node_embeds = self.encoder(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        batch_size = data.batch.max().item() + 1
        max_nodes = data.x.size(0) // batch_size if batch_size > 1 else data.x.size(0)

        reconstructed = self.reconstruction_decoder(node_embeds.X)

        return reconstructed

    def freeze_encoder(self):
        """Freeze the encoder weights for feature extraction."""
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()

    def unfreeze_encoder(self):
        """Unfreeze the encoder weights for fine-tuning."""
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = True
            encoder.train()

    def get_feature_dim(self):
        """Get the dimensionality of extracted features."""
        return (self.n_layers + 1) * self.hidden_dims  # Concatenated from all layers

    def apply_orthogonal_initialization(self):
        """Apply orthogonal random initialization to linear layers."""
        def orthogonal_init(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        for encoder in self.encoders:
            encoder.apply(orthogonal_init)

        for linear in self.linears_prediction:
            linear.apply(orthogonal_init)

        self.reconstruction_decoder.apply(orthogonal_init)

        print("Applied orthogonal initialization to GenGNNFeatureExtractor")
