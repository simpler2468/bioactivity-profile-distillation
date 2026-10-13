import dgl
import dgl.function as fn
import torch
from torch import nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training and input.dim() >= 2 and input.shape[0] <= 1:
            return input
        return super().forward(input)


def _get_graph_feature(graph, explicit_value, preferred_key, legacy_key):
    if explicit_value is not None:
        return explicit_value
    if preferred_key in graph.ndata:
        return graph.ndata[preferred_key]
    if preferred_key in graph.edata:
        return graph.edata[preferred_key]
    if legacy_key in graph.ndata:
        return graph.ndata[legacy_key]
    if legacy_key in graph.edata:
        return graph.edata[legacy_key]
    raise KeyError(f"Graph is missing feature key '{preferred_key}' or legacy key '{legacy_key}'.")


class EdgeAwareGINConv(nn.Module):
    """
    GIN-style message passing with edge features.

    Message: h_src + edge_embedding(edge_feat)
    Reduce: sum
    Update: MLP((1 + eps) * h_dst + neighbor_sum)
    """

    def __init__(self, hidden_dim, edge_dim=2, eps=0.0, train_eps=False):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, graph, node_feats, edge_feats):
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            graph.edata["e"] = self.edge_embed(edge_feats.float())
            graph.update_all(fn.u_add_e("h", "e", "m"), fn.sum("m", "neigh"))
            updated = (1.0 + self.eps) * node_feats + graph.ndata["neigh"]
            return self.mlp(updated)


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, graph, node_feats, atom_mask=None):
        with graph.local_scope():
            graph.ndata["h"] = node_feats
            weights = self.weight(node_feats)
            if atom_mask is not None:
                weights = weights * atom_mask.float().unsqueeze(-1)
            graph.ndata["w"] = weights
            return dgl.sum_nodes(graph, "h", weight="w")


class GINEncoder(nn.Module):
    def __init__(
        self,
        atom_dim=8,
        bond_dim=2,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        norm_type="batch_norm",
        readout="max",
    ):
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.readout_name = str(readout).lower()

        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.node_embed = nn.Linear(self.atom_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [EdgeAwareGINConv(self.hidden_dim, edge_dim=self.bond_dim) for _ in range(self.num_layers)]
        )

        norm_type = str(norm_type).lower()
        if norm_type in {"none", "null", "false"}:
            norm_type = "none"
        if norm_type not in {"batch_norm", "layer_norm", "none"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.norm_type = norm_type
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            if norm_type == "batch_norm":
                self.norms.append(SafeBatchNorm1d(self.hidden_dim))
            elif norm_type == "layer_norm":
                self.norms.append(nn.LayerNorm(self.hidden_dim))
            else:
                self.norms.append(nn.Identity())

        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attention_readout = AttentionReadout(self.hidden_dim)

    def forward_nodes(self, graph, atom_feat, bond_feat):
        hidden = self.node_embed(atom_feat.float())
        for conv, norm in zip(self.convs, self.norms):
            hidden = conv(graph, hidden, bond_feat)
            hidden = norm(hidden)
            hidden = self.activation(hidden)
            if self.dropout > 0:
                hidden = self.dropout_layer(hidden)
        return hidden

    def readout(self, graph, node_hidden, atom_mask=None):
        with graph.local_scope():
            graph.ndata["h"] = node_hidden
            if self.readout_name == "max":
                return dgl.max_nodes(graph, "h")
            if self.readout_name == "mean":
                return dgl.mean_nodes(graph, "h")
            if self.readout_name == "sum":
                return dgl.sum_nodes(graph, "h")
            if self.readout_name in {"attention", "weighted_sum"}:
                return self.attention_readout(graph, node_hidden, atom_mask)
        raise ValueError(f"Unsupported readout: {self.readout_name}")

    def forward(self, graph, atom_feat=None, bond_feat=None, atom_mask=None):
        atom_feat = _get_graph_feature(graph, atom_feat, "atom_feat", "node")
        bond_feat = _get_graph_feature(graph, bond_feat, "bond_feat", "edge")
        if atom_mask is None:
            if "atom_mask" in graph.ndata:
                atom_mask = graph.ndata["atom_mask"]
            elif "smask" in graph.ndata:
                atom_mask = graph.ndata["smask"]
        node_hidden = self.forward_nodes(graph, atom_feat, bond_feat)
        graph_hidden = self.readout(graph, node_hidden, atom_mask)
        return graph_hidden, node_hidden


class PlainGIN(nn.Module):
    """
    Plain multi-task GIN baseline.

    The output tensor is generic: classification tasks use it as logits, while
    regression tasks use it as values.
    """

    def __init__(
        self,
        output_dim,
        atom_dim=8,
        bond_dim=2,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        norm_type="batch_norm",
        readout="max",
        mlp_hidden_dims=None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.encoder = GINEncoder(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            readout=readout,
        )

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [hidden_dim]
        dims = [hidden_dim] + [int(dim) for dim in mlp_hidden_dims] + [self.output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        graph,
        atom_feat=None,
        bond_feat=None,
        atom_mask=None,
        return_hidden=False,
        return_dict=False,
    ):
        graph_hidden, node_hidden = self.encoder(graph, atom_feat, bond_feat, atom_mask)
        logits = self.head(graph_hidden)
        if return_dict:
            outputs = {"logits": logits}
            if return_hidden:
                outputs["hidden"] = graph_hidden
                outputs["node_hidden"] = node_hidden
            return outputs
        if return_hidden:
            return logits, graph_hidden
        return logits
