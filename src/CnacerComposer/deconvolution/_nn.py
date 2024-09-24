from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

activation_mapping = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

normalization_mapping = {"layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d}


class FClayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert activation in ["relu", "gelu", "tanh"]
        assert normalization in ["layernorm", "batchnorm"] or normalization is None
        assert 0 <= dropout <= 1

        activation_layer = activation_mapping.get(activation)
        normalization_layer = normalization_mapping.get(normalization, None)
        use_norm = False if normalization is None else True
        use_dropout = False if dropout == 0 else True

        self.layer = nn.Sequential(
            nn.Linear(n_in, n_out, bias=bias),
            activation_layer(),
            normalization_layer(n_out) if use_norm else nn.Identity(),
            nn.Dropout(dropout) if use_dropout else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
    
class MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int],
        activation: Literal["relu", "gelu", "tanh"] = "tanh",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        multiple_outputs: bool = False,
        norm_last_layer: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multiple_outputs = multiple_outputs
        
        dims = list(
            zip([n_in] + n_hidden, n_hidden + [n_out])
        )  # [(n_in, n_hidden_0), (n_hidden_0, n_hidden_1), (n_hidden_1, n_out)]

        self.layers = nn.ModuleList(
            [
                FClayer(
                    n_i,
                    n_o,
                    activation=activation,
                    normalization=normalization if i < len(dims) - 1 or norm_last_layer else None,
                    dropout=dropout,
                )
                for i,  (n_i, n_o) in enumerate(dims)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.multiple_outputs:
            output = []
            for layer in self.layers:
                x = layer(x)
                output.append(x)
            return output
        else:
            for layer in self.layers:
                x = layer(x)
            return x
        

class ResMLP(nn.Module):
    def __init__(
        self, 
        n_in: int,
        n_hidden: int,
        n_layers: int = 3,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.mlp = MLP(
            n_in=n_in,
            n_hidden=[n_hidden] * n_layers,
            n_out=n_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            multiple_outputs=True,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out[0] + out[-1]
    
class Encoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: list[int],
        n_mlp_layers: int,
        n_latent: int,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        last_k: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.last_k = last_k
        dims = list(
            zip([n_in] + n_hidden, n_hidden + [n_latent])
        )
         
        self.layers = nn.ModuleList([
            ResMLP(
                n_in=n_i,
                n_hidden=n_o,
                n_layers=n_mlp_layers,
                activation=activation,
                normalization=normalization,
                dropout=dropout
            ) for n_i, n_o in dims
        ])
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        output = []
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        return output[-1] if self.last_k == 1 else list(reversed(output[-self.last_k:]))


class PropDecoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: list[int],
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.1,
        last_k: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = MLP(
            n_in=n_in,
            n_out=n_in,
            n_hidden=n_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            norm_last_layer=False
        )
        
        self.last_k = last_k
        
        if self.last_k != 1:
            self.transform = MLP(
                n_in=last_k,
                n_out=1,
                n_hidden=[16, 16],
                activation="tanh",
                normalization="layernorm",
                norm_last_layer=False,
                dropout=0,
            )
        
    
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if self.last_k == 1:
            z_prop = x + self.mlp(x) # torch.relu(x + self.mlp(x))
        else:    
            x = torch.stack(x, dim=-1)
            bs = x.shape[0]
            x = x.reshape(-1, self.last_k)
            x = self.transform(x).reshape(bs, -1)
            z_prop = x + self.mlp(x)
        return z_prop / torch.sum(z_prop, dim=1, keepdim=True)
 
 
# class Conv1x1(nn.Module):
#     def __init__(
#         self,
#         n_in: int,
#         n_out: int,
#         use_norm: bool = True,
#         dropout: float = 0.2,
#         *args,
#         **kwargs
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.layers = nn.Sequential(
#             nn.Conv2d(n_in, n_out, kernel_size=1),
#             nn.ReLU(),
#             nn.InstanceNorm2d(n_out) if use_norm else nn.Identity(),
#             nn.Dropout(dropout)
#         )
             
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)
        
# class ExprDecoder(nn.Module):
#     def __init__(
#         self,
#         n_in: int,
#         n_hidden: list[int],
#         n_out: int,
#         dropout: float = 0.2,
#         last_k: int = 1,
#         *args,
#         **kwargs
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         dims = list(zip([n_in] + n_hidden, n_hidden + [n_out]))
        
        
#         self.layers = nn.ModuleList([
#             Conv1x1(
#                 n_in=n_i,
#                 n_out=n_o,
#                 use_norm=True if i < len(dims) - 1 else False,
#                 dropout=dropout if i < len(dims) - 1 else 0
#             )
#             for i, (n_i, n_o) in enumerate(dims)
#         ])
            
#         self.last_k = last_k
            
#     def forward(self, x: torch.Tensor | list[torch.Tensor]):
#         if self.last_k == 1:
#             x = x.permute(2, 1, 0)
#             for layer in self.layers:
#                 x = layer(x)
#             return x.permute(2, 1, 0)
#         else:
#             x = [x_.permute(2, 1, 0) for x_ in x]
#             feature = torch.zeros_like(x[0])
#             for x_, layer_ in zip(x, self.layers[:self.last_k]):
#                 x_ += feature
#                 feature = layer_(x_)
#             for layer in self.layers[self.last_k:]:
#                 feature = layer(feature)
#             return feature.permute(2, 1, 0)
    

class ExprDecoder(MLP):
    def __init__(
        self,
        n_latent: int,
        n_hidden: list[int],
        n_out: int,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        last_k: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            n_in=n_latent,
            n_hidden=n_hidden,
            n_out=n_out,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            multiple_outputs=True if last_k > 1 else False,
            norm_last_layer=False
        )
            
        self.last_k = last_k
        
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if self.last_k == 1:
            batch_size, n_labels, n_latent = x.shape
            x = x.reshape(batch_size * n_labels, n_latent)
            for layer in self.layers:
                x = layer(x)
            return x.reshape(batch_size, n_labels, -1)
        else:
            batch_size, n_labels, _ = x[0].shape
            x = [item.reshape(batch_size * n_labels, -1) for item in x]
            feature = torch.zeros_like(x[0])
            for x_, layer_ in zip(x, self.layers[:self.last_k]):
                x_ += feature
                feature = layer_(feature)
            for layer_ in self.layers[self.last_k:]:
                feature = layer_(feature)
            return feature.reshape(batch_size, n_labels, -1)


class Multiplier(nn.Module):
    def __init__(
        self,
        activation: Literal["relu", "tanh"] | None = None,
        last_k: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert activation in ["relu", "tanh"] or activation is None
        if activation is None:
            self.activation_layer = nn.Identity()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        elif activation == "tanh":
            self.activation_layer = nn.Tanh()
            
        self.last_k = last_k
        
    def forward(
        self, 
        bulk: torch.Tensor | list[torch.Tensor], 
        reference: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:

        return self.activation_layer(bulk @ reference.t()) if self.last_k == 1 \
            else [self.activation_layer(b @ r.t()) for b, r in zip(bulk, reference)]

class DotProductor(nn.Module):
    def __init__(
        self,
        activation: Literal["relu", "tanh"] | None = None,
        last_k: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        assert activation in ["relu", "tanh"] or activation is None
        if activation is None:
            self.activation_layer = nn.Identity()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        elif activation == "tanh":
            self.activation_layer = nn.Tanh()
            
        self.last_k = last_k
        
    def forward(
        self, 
        bulk: torch.Tensor | list[torch.Tensor], 
        reference: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if self.last_k == 1:
            expr = self.activation_layer(bulk.unsqueeze(1) * reference.unsqueeze(0)) # (bs, 1, latent) * (1, ct, latent)
            return expr
        else:
            output = [
                self.activation_layer(b.unsqueeze(1) * r.unsqueeze(0)) \
                    for b, r in zip(bulk, reference)
            ]
            return output


class Deconv(nn.Module):
    def __init__(
        self,
        n_labels: int,
        n_genes: int,
        n_hidden: list[int],
        n_mlp_layers: int,
        n_latent: int,
        activation: Literal["relu", "gelu"] = "relu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        last_k: int = 1,
        task: Literal["prop", "expr", "both"] = "both",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.reference_encoder = Encoder(
            n_in=n_genes,
            n_hidden=n_hidden,
            n_mlp_layers=n_mlp_layers,
            n_latent=n_latent,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            last_k=last_k
        )
        
        self.bulk_encoder = Encoder(
            n_in=n_genes,
            n_hidden=n_hidden,
            n_mlp_layers=n_mlp_layers,
            n_latent=n_latent,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            last_k=last_k
        )
        self.multiplier = Multiplier(
            activation="relu",
            last_k=last_k
        )

        self.dot_productor = DotProductor(
            activation="relu",
            last_k=last_k
        )
        
        self.proportion_decoder = PropDecoder(
            n_in=n_labels,
            n_hidden=[128, 128],
            activation="relu",
            normalization=normalization,
            dropout=0,
            last_k=last_k
        )
        
        self.expression_decoder = ExprDecoder(
            n_latent=n_latent,
            n_hidden=n_hidden[::-1],
            n_out=n_genes,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            last_k=last_k
        )
            
        self.init_params()
        
        self.n_genes, self.n_labels = n_genes, n_labels
        self.n_latent = n_latent
        
        self.task = task
        
    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                layer.bias.data.fill_(0.01)
                
    def prop_forward(self, z_bulk, z_ref) -> torch.Tensor:
        z_prop = self.multiplier(z_bulk, z_ref)
        prop = self.proportion_decoder(z_prop)
        return prop
    
    def expr_forward(self, z_bulk, z_ref) -> torch.Tensor:
        z_expr = self.dot_productor(z_bulk, z_ref)
        expr = self.expression_decoder(z_expr)
        return expr 
               
    def forward(
        self, 
        bulk: torch.Tensor, 
        reference: torch.Tensor | None = None
    ) -> dict[Literal["prop", "expr"], torch.Tensor | None]:
        z_ref = self.reference_encoder(reference)  # ct x n_hidden
        z_bulk = self.bulk_encoder(bulk)  # batch x n_hidden
        
        output = {
            "prop": None,
            "expr": None,
        }
        
        if self.task == "prop":   
            output["prop"] = self.prop_forward(z_bulk, z_ref)
        elif self.task == "expr":
            output["expr"] = self.expr_forward(z_bulk, z_ref)
        else:
            output["prop"] = self.prop_forward(z_bulk, z_ref)
            output["expr"] = self.expr_forward(z_bulk, z_ref)
        
        return output        
            
    def loss_prop(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(preds, target)
    
    def loss_expr(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(preds, targets)
        
    def loss(
        self,
        preds: dict[Literal["prop", "expr"], torch.Tensor | None], 
        targets: dict[Literal["prop", "expr"], torch.Tensor | None]
    ) -> dict[Literal["prop", "expr"], torch.Tensor]:
        output = {
            "loss_prop": None,
            "loss_expr": None
        }
        
        if self.task == "prop":
            output["loss_prop"] = self.loss_prop(preds["prop"], targets["prop"])
        elif self.task == "expr":
            output["loss_expr"] = self.loss_expr(preds["expr"], targets["expr"])
        else:
            output["loss_prop"] = self.loss_prop(preds["prop"], targets["prop"])
            output["loss_expr"] = self.loss_expr(preds["expr"], targets["expr"])
        
        return output