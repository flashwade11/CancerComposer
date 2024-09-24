from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

activation_mapping = {"relu": nn.ReLU, "gelu": nn.GELU}

normalization_mapping = {"layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d}


class FClayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert activation in ["relu", "gelu"]
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
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        dropout: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dims = list(
            zip([n_in] + n_hidden, n_hidden + [n_out])
        )  # [(n_in, n_hidden_0), (n_hidden_0, n_hidden_1), (n_hidden_1, n_out)]

        self.layers = nn.Sequential(
            *[
                FClayer(
                    n_i,
                    n_o,
                    activation=activation,
                    normalization=normalization,
                    dropout=dropout,
                )
                for n_i, n_o in dims
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_head: int = 12,
        n_layers: int = 3,
        dim_feedforward: int = 1024,
        activation: Literal["relu", "gelu"] = "gelu",
        dropout: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_head,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)


class DownSampling(nn.Module):
    def __init__(
        self,
        n_in,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] | None = "layernorm",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.linear = FClayer(
            n_in,
            n_in // 2,
            activation=activation,
            normalization=normalization,
            dropout=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_head: int = 12,
        n_layers: int = 3,
        dim_feedforward: int = 1024,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        down_sampling: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.transformer_encoder = TransformerEncoder(
            n_hidden=n_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
        )
        
        self.down_sampling = DownSampling(n_hidden, activation=activation, normalization=normalization) \
            if down_sampling else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.transformer_encoder(x) + x
        # z = self.transformer_encoder(x)
        return z, self.down_sampling(z)

class Encoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        mlp_hidden: list[int],
        mlp_out: int,
        transformer_hidden: list[int],
        n_head: int = 12,
        n_transformer_layers: int = 3,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        return_last: bool = True,
        *args,
        **kwargs,
    ) -> None:
        assert mlp_out == transformer_hidden[0]
        super().__init__(*args, **kwargs)
        
        self.return_last = return_last
        
        self.mlp = MLP(
            n_in=n_in,
            n_out=mlp_out,
            n_hidden=mlp_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                n_hidden_,
                n_head=n_head,
                dim_feedforward=n_hidden_,
                n_layers=n_transformer_layers,
                activation=activation,
                normalization=normalization,
                dropout=dropout,
                down_sampling=True if i < len(transformer_hidden) - 1 else False,
            )
            for i, n_hidden_ in enumerate(transformer_hidden)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.mlp(x)
        
        output = []
        for block in self.transformer_blocks:
            z, x = block(x)
            output.append(z)
        
        return output[-1] if self.return_last else output
        
        
class PropDecoder(nn.Module):
    _weights = {
        2: [0.7, 0.3],
        3: [0.6, 0.3, 0.1],
        4: [0.4, 0.3, 0.2, 0.1]
    }
    def __init__(
        self,
        n_in: int,
        n_hidden: list[int],
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.1,
        use_deepest: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_deepest = use_deepest
        self.mlp = MLP(
            n_in=n_in,
            n_out=n_in,
            n_hidden=n_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if self.use_deepest:
            return x + self.mlp(x)
        else:
            weights = self._weights[len(x)]
            weight_x = [t * w for t, w in zip(x, weights)]
            x = torch.sum(torch.stack(weight_x), dim=0)
            return x + self.mlp(x)
          
class ExprDecoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        mlp_hidden: list[int],
        transformer_hidden: list[int],
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        use_deepest: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_deepest = use_deepest
        self.mlp = MLP(
            n_in=n_in,
            n_out=n_out,
            n_hidden=mlp_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        
        linear_hidden = tuple(zip(transformer_hidden[:-1], transformer_hidden[1:]))
        self.linears = nn.ModuleList([FClayer(n_i, n_o) for n_i, n_o in linear_hidden])
        
    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if self.use_deepest:
            for linear in self.linears:
                x = linear(x)
            return self.mlp(x)
        else:
            feature = torch.zeros_like(x[-1])
            for z, linear in (zip(x[-1:0:-1], self.linears)):
                feature = linear(z + feature)
            feature = feature + x[0]
            return self.mlp(feature)    


class Multiplier(nn.Module):
    def __init__(
        self,
        activation: Literal["relu", "tanh"] | None = None,
        use_deepest: bool = True,
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
            
        self.use_deepest = use_deepest
        
    def forward(
        self, 
        bulk: torch.Tensor | list[torch.Tensor], 
        reference: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        if self.use_deepest:
            return self.activation_layer(bulk @ reference.t())
        else:
            return [self.activation_layer(b @ r.t()) for b, r in zip(bulk, reference)]
    
    
class DotProductor(nn.Module):
    def __init__(
        self,
        activation: Literal["relu", "tanh"] | None = None,
        use_deepest: bool = False,
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
            
        self.use_deepest = use_deepest
        
    def forward(
        self, 
        bulk: torch.Tensor | list[torch.Tensor], 
        reference: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_deepest:
            n_hidden = bulk.shape[1]
            expr = self.activation_layer(bulk.unsqueeze(1) * reference.unsqueeze(0)) # (bs, 1, hidden) * (1, ct, hidden)
            return expr.reshape(-1, n_hidden) # (bs * ct, hidden)
        else:
            output = []
            for b, r in zip(bulk, reference):
                n_hidden = b.shape[1]
                expr = self.activation_layer(b.unsqueeze(1) * r.unsqueeze(0)) # (bs, 1, hidden) * (1, ct, hidden)
                output.append(expr.reshape(-1, n_hidden))
            return output
       

class Deconv(nn.Module):
    def __init__(
        self,
        n_labels: int,
        n_genes: int,
        mlp_hidden: list[int],
        mlp_out: int,
        transformer_hidden: list[int],
        n_hidden: int,
        n_head: int = 8,
        n_transformer_layers: int = 3,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        dropout_prop: float = 0.1,
        use_deepest: bool = False,
        only_prop: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_encoder = Encoder(
            n_in=n_genes,
            mlp_hidden=mlp_hidden,
            mlp_out=mlp_out,
            transformer_hidden=transformer_hidden,
            n_head=n_head,
            n_transformer_layers=n_transformer_layers,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            return_last=use_deepest
        )
        
        self.proportion_decoder = PropDecoder(
            n_in=n_labels,
            n_hidden=[128, 128],
            activation=activation,
            normalization=normalization,
            dropout=dropout_prop,
            use_deepest=use_deepest
        )
        self.multiplier = Multiplier(
            activation="relu",
            use_deepest=use_deepest
        )
        
        if not only_prop:
            self.bulk_encoder = Encoder(
                n_in=n_genes,
                mlp_hidden=mlp_hidden,
                mlp_out=mlp_out,
                transformer_hidden=transformer_hidden,
                n_head=n_head,
                n_transformer_layers=n_transformer_layers,
                activation=activation,
                normalization=normalization,
                dropout=dropout,
                return_last=use_deepest
            )

            self.expression_decoder = ExprDecoder(
                n_in=mlp_out,
                n_out=n_genes,
                mlp_hidden=mlp_hidden[::-1],
                transformer_hidden=transformer_hidden[::-1],
                activation=activation,
                normalization=normalization,
                dropout=dropout,
                use_deepest=use_deepest
            )

            
            
            
            self.dot_productor = DotProductor(
                activation="relu",
                use_deepest=use_deepest
            )
    

        # self.register_buffer("z_ref", self.reference_encoder(torch.zeros(n_labels, n_genes)))
        self.init_params()

        self.n_genes, self.n_labels = n_genes, n_labels
        self.n_hidden = n_hidden
        self.only_prop = only_prop

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, bulk: torch.Tensor, reference: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # batch_size = bulk.shape[0]

        # if reference is not None:
        #     z_ref = self.reference_encoder(reference)  # ct x n_hidden
        #     self.z_ref = z_ref.detach()
        # else:
        #     z_ref = self.z_ref
        z_ref = self.reference_encoder(reference)  # ct x n_hidden
        z_bulk = self.bulk_encoder(bulk)  # batch x n_hidden
        # z_prop = self.multiplier(z_bulk, z_ref)
        # z_prop = self.proportion_decoder(z_prop)
        # prop = z_prop / torch.sum(z_prop, dim=1, keepdim=True)
        prop = self.prop_forward(z_ref, z_bulk)
        if not self.only_prop:
            expr = self.expr_forward(z_bulk, z_ref)
        # z_expr = self.dot_productor(z_bulk, z_ref)
        # z_expr = self.expression_decoder(z_expr)
        # expr = z_expr.reshape(batch_size, self.n_labels, self.n_genes)

        return {"prop": prop} if self.only_prop else {"prop": prop, "expr": expr} 

    def prop_forward(self, z_ref, z_bulk):
        z_prop = self.multiplier(z_bulk, z_ref)
        z_prop = self.proportion_decoder(z_prop)
        prop = z_prop / torch.sum(z_prop, dim=1, keepdim=True)
        return prop
    
    def expr_forward(self, z_bulk, z_ref):
        z_expr = self.dot_productor(z_bulk, z_ref)
        z_expr = self.expression_decoder(z_expr)
        expr = z_expr.reshape(z_bulk.shape[0], self.n_labels, self.n_genes)
        return expr
    
    def loss(
        self,
        alpha: float,
        pred_prop: torch.Tensor,
        pred_expr: torch.Tensor,
        true_prop: torch.Tensor,
        true_expr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_prop = F.mse_loss(pred_prop, true_prop)
        if not self.only_prop:
            loss_expr = F.mse_loss(pred_expr, true_expr)
            loss_total = alpha * loss_prop + loss_expr
        return {"loss_prop": loss_prop} if self.only_prop \
            else {
                "loss_prop": loss_prop,
                "loss_expr": loss_expr,
                "loss_total": loss_total,
            }    


def init_model(name: Literal["model1", "model2"], n_labels: int, n_genes: int, *args, **kwargs):
    models = {
        "model1": {
            "n_labels": n_labels,
            "n_genes": n_genes,
            "mlp_hidden": [],
            "mlp_out": 768 * 8,
            "transformer_hidden": [768 * 8, 768 * 4, 768 * 2],
            "n_hidden": 768,
            "n_head": 12,
            "n_transformer_layers": 3,
            "activation": "gelu",
            "normalization": "layernorm",
            "dropout": 0.2,
            "dropout_prop": 0.1,
        },
        "model2": {
            "n_labels": n_labels,
            "n_genes": n_genes,
            "mlp_hidden": [8192, 4096],
            "mlp_out": 2048,
            "transformer_hidden": [2048, 1024],
            "n_hidden": 512,
            "n_head": 8,
            "n_transformer_layers": 3,
            "activation": "gelu",
            "normalization": "layernorm",
            "dropout": 0.2,
            "dropout_prop": 0.1,
        },
        "neomodel": {
            "n_labels": n_labels,
            "n_genes": n_genes,
            "mlp_hidden": [8192, 2048, 512],
            "mlp_out": 512,
            "transformer_hidden": [512, 256, 128],
            "n_hidden": 128,
            "n_head": 8,
            "n_transformer_layers": 3,
            "activation": "gelu",
            "normalization": "layernorm",
            "dropout": 0.2,
            "dropout_prop": 0.1,
            "use_deepest": False,
        },
        "large_neo_model": {
            "n_labels": n_labels,
            "n_genes": n_genes,
            "mlp_hidden": [8192, 4096, 2048, 1024],
            "mlp_out": 1024,
            "transformer_hidden": [1024, 512, 256],
            "n_hidden": 256,
            "n_head": 8,
            "n_transformer_layers": 3,
            "activation": "gelu",
            "normalization": "layernorm",
            "dropout": 0.2,
            "dropout_prop": 0.1,
            "use_deepest": False,
        },
        "model3": {
            "n_labels": n_labels,
            "n_genes": n_genes,
            "mlp_hidden": [8192, 4096, 2048],
            "mlp_out": 2048,
            "transformer_hidden": [2048, 1024, 512],
            "n_hidden": 512,
            "n_head": 8,
            "n_transformer_layers": 3,
            "activation": "gelu",
            "normalization": "layernorm",
            "dropout": 0.2,
            "dropout_prop": 0.1,
            "use_deepest": False,   
        }
    }

    assert name in models.keys(), f"Invalid model name: {name}"
    # if name in ["model1", "model2"]:
    #     return BulkDeconv(**models[name])
    # else:
    #     return Deconv(**models[name])
    return Deconv(**models[name])



class PropDeconv(nn.Module):
    def __init__(
        self,
        n_labels: int,
        n_genes: int,
        mlp_hidden: list[int],
        n_hidden: int,
        activation: Literal["relu", "gelu"] = "gelu",
        normalization: Literal["layernorm", "batchnorm"] = "layernorm",
        dropout: float = 0.2,
        dropout_prop: float = 0.1,
        use_deepest: bool = False,
        only_prop: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # self.reference_encoder = Encoder(
        #     n_in=n_genes,
        #     mlp_hidden=mlp_hidden,
        #     mlp_out=mlp_out,
        #     transformer_hidden=transformer_hidden,
        #     n_head=n_head,
        #     n_transformer_layers=n_transformer_layers,
        #     activation=activation,
        #     normalization=normalization,
        #     dropout=dropout,
        #     return_last=use_deepest
        # )
        
        self.reference_encoder = MLP(
            n_in=n_genes,
            n_out=n_hidden,
            n_hidden=mlp_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        
        self.bulk_encoder = MLP(
            n_in=n_genes,
            n_out=n_hidden,
            n_hidden=mlp_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        
        self.proportion_decoder = PropDecoder(
            n_in=n_labels,
            n_hidden=[128, 128],
            activation=activation,
            normalization=normalization,
            dropout=dropout_prop,
            use_deepest=use_deepest
        )
        self.multiplier = Multiplier(
            activation="relu",
            use_deepest=use_deepest
        )

        # self.register_buffer("z_ref", self.reference_encoder(torch.zeros(n_labels, n_genes)))
        self.init_params()

        self.n_genes, self.n_labels = n_genes, n_labels
        self.n_hidden = n_hidden
        self.only_prop = only_prop

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, bulk: torch.Tensor, reference: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # batch_size = bulk.shape[0]

        # if reference is not None:
        #     z_ref = self.reference_encoder(reference)  # ct x n_hidden
        #     self.z_ref = z_ref.detach()
        # else:
        #     z_ref = self.z_ref
        z_ref = self.reference_encoder(reference)  # ct x n_hidden
        z_bulk = self.bulk_encoder(bulk)  # batch x n_hidden
        prop = self.prop_forward(z_bulk, z_ref)
        # z_expr = self.dot_productor(z_bulk, z_ref)
        # z_expr = self.expression_decoder(z_expr)
        # expr = z_expr.reshape(batch_size, self.n_labels, self.n_genes)

        return {"prop": prop}

    def prop_forward(self, z_bulk, z_ref):
        z_prop = self.multiplier(z_bulk, z_ref)
        z_prop = self.proportion_decoder(z_prop)
        prop = z_prop / torch.sum(z_prop, dim=1, keepdim=True)
        return prop
    
    def loss(
        self,
        pred_prop: torch.Tensor,
        true_prop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_prop = F.l1_loss(pred_prop, true_prop)
        return {"loss_prop": loss_prop}