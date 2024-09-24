from pathlib import Path
from typing import Literal

import anndata as ad
import lightning as L
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import ConcordanceCorrCoef, MetricCollection, PearsonCorrCoef

from ._dataset import load_dataset_from_anndata, load_dataset_from_files
from ._nn import Deconv

from lightning.pytorch.utilities.parsing import AttributeDict


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        bulk_dict: dict[str, ad.AnnData] | None = None,
        data_folder: str | Path | None = None,
        file_name_dict: dict[Literal["train", "val", "test", "predict"], list[str]] | None = None,
        # layer: str,
        *args,
        n_top_genes: int = 5000,
        batch_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        self.bulk_dict = bulk_dict
        self.data_folder = data_folder
        self.file_name_dict = file_name_dict
        self.n_top_genes = n_top_genes
        self.batch_size = batch_size

    def _load_dataset(
        self,
        stage: Literal["fit", "test", "predict"],
        _type: Literal["train", "val", "test", "predict"]
    ):
        if self.bulk_dict:
            return load_dataset_from_anndata(
                adata_list=self.bulk_dict[_type],
                n_top_genes=self.n_top_genes,
                stage=stage
            )
        if self.data_folder and self.file_name_dict:
            return load_dataset_from_files(
                folder=self.data_folder,
                file_name_list=self.file_name_dict[_type],
                n_top_genes=self.n_top_genes,
                stage=stage
            )
        
    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._load_dataset(
                stage=stage,
                _type="train"
            )
            self.val_dataset = self._load_dataset(
                stage=stage,
                _type="val"
            )
        if stage == "test":
            self.test_dataset = self._load_dataset(
                stage=stage,
                _type="test"
            )
        if stage == "predict":
            self.predict_dataset = self._load_dataset(
                stage=stage,
                _type="predict"
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_params: dict,
        reference_cell_type_expr: pd.DataFrame | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if reference_cell_type_expr is not None:
            assert model_params["n_labels"], model_params["n_genes"] == reference_cell_type_expr.shape
            
        self.reference = reference_cell_type_expr
        self.alpha = model_params.pop("alpha") if "alpha" in model_params else 20
        self.task = model_params["task"]
        
        self.model_params = AttributeDict(model_params)
        
        optimizer_params = model_params.pop("optimizer", {})
        self.optimizer_params = {
            "lr": 1e-4,
            "lr_factor":0.1,
            "scheduler_patience": 50,
            "min_lr": 1e-5,
            "monitor": "val_loss" if self.task == "both" else "val_loss_prop"
        }
        
        self.optimizer_params.update(optimizer_params)
        self.optimizer_params = AttributeDict(self.optimizer_params)
        
        self.model = Deconv(**model_params)
              
    def setup(self, stage: str) -> None:
        metrics = MetricCollection(
            [
                PearsonCorrCoef(num_outputs=self.hparams.model_params["n_labels"]),
                ConcordanceCorrCoef(num_outputs=self.hparams.model_params["n_labels"]),
            ]
        )

        self.train_metric_collection = metrics.clone(prefix="train_")
        self.val_metric_collection = metrics.clone(prefix="val_")

        self.reference = (
            torch.from_numpy(self.reference.values).float() \
                if isinstance(self.reference, pd.DataFrame) else None
        )
        
    def forward(
        self, 
        bulk: torch.Tensor, 
        reference: torch.Tensor | None = None,
        *args,
        **kwargs
    ) -> dict[Literal["prop", "expr"], torch.Tensor | None]:
        return self.model(bulk, reference)

    def forward_step(self, batch, stage: Literal["train", "val"], *args, **kwargs) -> dict[str, torch.Tensor]:
        bulk, prop, expr = batch
        self.reference = self.reference.to(bulk)
        preds = self(bulk, self.reference)
        loss = self.model.loss(
            preds=preds,
            targets=dict(prop=prop, expr=expr)
        )

        loss = {f"{stage}_{k}": v for k, v in loss.items()}
        if self.task == "both":
            loss[f"{stage}_loss"] = loss[f"{stage}_loss_prop"] * self.alpha + loss[f"{stage}_loss_expr"]
            
        self.log_dict(loss, on_step=False, on_epoch=True, prog_bar=True)

        if stage == "train":
            self.train_metric_collection(preds["prop"].to(torch.float64), prop.to(torch.float64))
        elif stage == "val":
            self.val_metric_collection(preds["prop"].to(torch.float64), prop.to(torch.float64))
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        return loss
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.forward_step(batch, stage="train")
        if self.task == "prop":
            return loss["train_loss_prop"]
        elif self.task == "expr":
            return loss["train_loss_expr"]
        else:
            return loss["train_loss"]
    
    def on_train_epoch_end(self) -> None:
        metrics = self.train_metric_collection.compute()
        for name, value in metrics.items():
            self.log(name, value.mean(), prog_bar=True)
        self.train_metric_collection.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.forward_step(batch, stage="val")
        if self.task == "prop":
            return loss["val_loss_prop"]
        elif self.task == "expr":
            return loss["val_loss_expr"]
        else:
            return loss["val_loss"]

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric_collection.compute()
        for name, value in metrics.items():
            self.log(name, value.mean(), prog_bar=True)
        self.val_metric_collection.reset()

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def predict_step(self, batch) -> dict[str, torch.Tensor]:
        bulk = batch
        preds = self(bulk, self.reference.to(bulk))
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer_params["lr"]
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.optimizer_params["lr_factor"],
            patience=self.optimizer_params["scheduler_patience"],
            min_lr=self.optimizer_params["min_lr"],
            threshold=1e-3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor":self.optimizer_params["monitor"]
            },
        }