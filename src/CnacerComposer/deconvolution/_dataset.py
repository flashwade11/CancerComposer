import gc
from pathlib import Path
from typing import Literal

import anndata as ad
import torch
from torch.utils.data import ConcatDataset, Dataset


class Anndataset(Dataset):
    def __init__(
        self, 
        adata: ad.AnnData, 
        n_top_genes: int,
        stage: Literal["fit", "test", "predict"] = "fit",
        precision: Literal["float", "half"] = "half"
    ) -> None:
        super().__init__()
        assert stage in ["fit", "test", "predict"]
    
        self.stage = stage
            
        if stage in ["fit", "test"]:
            self.bulk = _change_precision(torch.from_numpy(adata.obsm[f"X_{n_top_genes}_MMS"]), precision=precision)
            self.prop = _change_precision(torch.from_numpy(adata.obsm["prop"].to_numpy()), precision=precision)
            self.expr = _change_precision(torch.from_numpy(adata.uns[f"expr_{n_top_genes}_log1p"]), precision=precision)
            assert self.bulk.shape[0] == self.prop.shape[0] == self.expr.shape[0]
        elif stage == "predict":
            self.bulk = _change_precision(torch.from_numpy(adata.obsm[f"X_{n_top_genes}_MMS"]), precision=precision)
    
    def __len__(self) -> int:
        return self.bulk.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.stage in ["fit", "test"]:
            return self.bulk[index], self.prop[index], self.expr[index]
        elif self.stage == "predict":
            return self.bulk[index]
    
    @classmethod
    def from_file(
        cls,
        file: str | Path,
        n_top_genes: int,
        stage: Literal["fit", "test", "predict"] = "fit",
        precision: Literal["float", "half"] = "half"
    ):
        adata = ad.read_h5ad(file)
        obj = cls(adata, n_top_genes, stage=stage, precision=precision)
        del adata
        gc.collect()
        return obj
        
def load_dataset_from_anndata(
    adata_list: list[ad.AnnData], 
    n_top_genes: int,
    stage: Literal["fit", "test", "predict"] = "fit"    
) -> Anndataset:
    return ConcatDataset([Anndataset(adata=adata, n_top_genes=n_top_genes, stage=stage) for adata in adata_list])

def load_dataset_from_files(
    folder: str | Path, 
    file_name_list: list[str], 
    n_top_genes: int,
    stage: Literal["fit", "test", "predict"] = "fit",
    precision: Literal["float", "half"] = "half"
) -> Anndataset:
    if isinstance(folder, str):
        folder = Path(folder)
    for file_name in file_name_list:
        file_path = folder.joinpath(file_name)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")
    
    return ConcatDataset([Anndataset.from_file(file=folder.joinpath(file_name), n_top_genes=n_top_genes, stage=stage, precision=precision) for file_name in file_name_list])

def _change_precision(t: torch.Tensor, precision: Literal["float", "half"]) -> torch.Tensor:
    if precision == "float":
        return t.float()
    elif precision == "half":
        return t.half()
    else:
        raise ValueError("Precision can only be `float` or `half`")
    
    