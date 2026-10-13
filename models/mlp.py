import torch
from torch import nn


def build_hidden_dims(
    input_dim,
    output_dim,
    num_hidden_layers=3,
    shrink_ratio=0.5,
    min_hidden_dim=None,
):
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    num_hidden_layers = int(num_hidden_layers)
    shrink_ratio = float(shrink_ratio)

    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}.")
    if not 0 < shrink_ratio <= 1:
        raise ValueError(f"shrink_ratio must be in (0, 1], got {shrink_ratio}.")

    if min_hidden_dim is None:
        min_hidden_dim = max(64, output_dim * 2)
    min_hidden_dim = int(min_hidden_dim)
    if min_hidden_dim <= 0:
        raise ValueError(f"min_hidden_dim must be positive, got {min_hidden_dim}.")

    hidden_dims = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        next_dim = max(min_hidden_dim, int(round(current_dim * shrink_ratio)))
        hidden_dims.append(next_dim)
        current_dim = next_dim
    return hidden_dims


def make_activation(activation="leaky_relu", negative_slope=0.01):
    activation = str(activation).lower()
    if activation == "relu":
        return nn.ReLU()
    if activation in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=negative_slope)
    if activation == "gelu":
        return nn.GELU()
    if activation == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {activation}")


class TabularMLP(nn.Module):
    """
    Multi-task MLP for fingerprint/descriptor matrices.

    The output head is intentionally generic: classification tasks interpret
    outputs as logits, while regression tasks interpret outputs as values.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_layers=3,
        shrink_ratio=0.5,
        hidden_dims=None,
        min_hidden_dim=None,
        dropout=0.2,
        activation="leaky_relu",
        norm_type="layer_norm",
        batch_norm=False,
        negative_slope=0.01,
    ):
        super().__init__()
        if not 0 <= float(dropout) < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.activation_name = activation
        self.negative_slope = float(negative_slope)

        if hidden_dims is None:
            hidden_dims = build_hidden_dims(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                num_hidden_layers=num_hidden_layers,
                shrink_ratio=shrink_ratio,
                min_hidden_dim=min_hidden_dim,
            )
        self.hidden_dims = [int(dim) for dim in hidden_dims]

        if norm_type is None:
            norm_type = "batch_norm" if batch_norm else "none"
        norm_type = str(norm_type).lower()
        if norm_type in {"none", "null", "false"}:
            norm_type = "none"
        if norm_type not in {"batch_norm", "layer_norm", "none"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        self.norm_type = norm_type

        layers = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if self.norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(make_activation(self.activation_name, self.negative_slope))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous_dim, self.output_dim)
        self.reset_parameters()

    @property
    def architecture(self):
        return [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nonlinearity = (
                    "leaky_relu"
                    if self.activation_name.lower() in {"leaky_relu", "leakyrelu"}
                    else "relu"
                )
                a = self.negative_slope if nonlinearity == "leaky_relu" else 0
                nn.init.kaiming_uniform_(module.weight, a=a, nonlinearity=nonlinearity)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_hidden=False, return_dict=False):
        x = x.float()
        hidden = self.backbone(x)
        logits = self.output_layer(hidden)

        if return_dict:
            outputs = {"logits": logits}
            if return_hidden:
                outputs["hidden"] = hidden
            return outputs
        if return_hidden:
            return logits, hidden
        return logits

    def extra_repr(self):
        return (
            f"architecture={self.architecture}, dropout={self.dropout}, "
            f"activation={self.activation_name}, norm_type={self.norm_type}"
        )
