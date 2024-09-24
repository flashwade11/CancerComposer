import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import anndata as ad 
from sklearn.preprocessing import minmax_scale
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from .deconvolution._lightning_module import ModelModule

CancerComposer_path = Path(os.getenv("CancerComposer_FOLDER", default = str(Path.home() / ".CancerComposer")))


class Model:
    _model_cls = ModelModule
    
    def __init__(self, model: ModelModule, device: Literal["cpu", "gpu"] = "gpu") -> None:
        self.model = model
        self.reference = self.model.reference.copy(deep=True)
        
        self.trainer = pl.Trainer(enable_checkpointing=False, logger=False, accelerator=device)
        
    
    @classmethod
    def load(
        cls, 
        organ: str | None = None,
        ckpt_path: str | Path | None = None,
        map_location: str = "cpu",
        device: Literal["cpu", "gpu"] = "gpu",
    ):
        if organ is None and ckpt_path is None:
            raise ValueError("Error!")
        if ckpt_path is None:
            ckpt_path = CancerComposer_path.joinpath(f"{organ}.ckpt")
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileExistsError("Please check your checkpoint file wether in the ~/.CancerComposer folder")
        obj = cls(model=cls._model_cls.load_from_checkpoint(ckpt_path, map_location=map_location), device=device)
        obj.organ = organ
        return obj
    
    @property
    def cell_types(self) -> pd.Index:
        return self.reference.index
    
    @property
    def top_genes(self) -> pd.Index:
        return self.reference.columns
    
    def predict(self, bulk: pd.DataFrame):
        bulk_dataset = _prepare_train_bulkdataset(bulk, top_genes=self.top_genes)
        bulk_dataloader = DataLoader(bulk_dataset, batch_size=len(bulk_dataset), shuffle=False)
        results = self.trainer.predict(model=self.model, dataloaders=bulk_dataloader)[0]
        
        predict_proportion = pd.DataFrame(results["prop"].cpu().numpy(), columns=self.cell_types)
        
        predict_expression = list(map(
            lambda x: pd.DataFrame(x.cpu().numpy(), index=self.cell_types, columns=self.top_genes).apply(np.expm1), 
            results["expr"]
        ))
        
        return predict_proportion, predict_expression
    
    def finetune(
        self, 
        reference: ad.AnnData,
        cell_type_key: str = "cell_type",
    ):
        if cell_type_key not in reference.obs_keys():
            _cell_typist_annotate(reference, self.organ)
        else:
            n_cell_types = reference.obs[cell_type_key].nunique()
            if n_cell_types == ...:
                pass
            
        bulk = _prepare_finetune_bulkdataset(reference, n_sample=5000)
        
            


class Bulkdataset(Dataset):
    def __init__(self, bulk: np.ndarray) -> None:
        super().__init__()
        self.bulk = torch.from_numpy(bulk).float()
        
    def __len__(self) -> int:
        return self.bulk.shape[0]
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.bulk[index]
    
    
def _cell_typist_annotate():
    raise NotImplementedError

def _simulate_bulk(*args, **kwargs):
    raise NotImplementedError


def _align_genes(bulk: pd.DataFrame, top_genes: pd.Index) -> pd.DataFrame:
    genes = bulk.columns
    missing_genes = top_genes.difference(genes)
    bulk[missing_genes] = 0.0
    bulk = bulk[top_genes]
    return bulk

def _prepare_train_bulkdataset(bulk: pd.DataFrame, top_genes: pd.Index) -> Bulkdataset:
    _bulk = bulk.copy(deep=True)
    _bulk = _align_genes(_bulk, top_genes)
    _bulk = minmax_scale(_bulk, axis=1)
    return Bulkdataset(_bulk)
    
def _prepare_finetune_bulkdataset(top_genes: np.ndarray, *args, **kwargs):
    _bulk = _simulate_bulk()
    _bulk = _align_genes(_bulk, top_genes=top_genes)
    _bulk = minmax_scale(_bulk, axis=1)
    return Bulkdataset(_bulk)

def _preprocess_finetune_reference(reference: ad.AnnData, top_genes: pd.Index):
    raise NotImplementedError
