from pathlib import Path
from typing import Literal

import anndata as ad
import joblib
import pandas as pd
import scanpy as sc


def _reference_calculate_cell_type_expression(
    adata: ad.AnnData,
    label_key: str,
    *,
    layer: str | None = "counts",
    agg: Literal["mean", "sum"] = "mean",
    n_top_genes: int = 20000,
) -> None | dict[str, dict | pd.DataFrame]:
    assert label_key in adata.obs_keys()
    assert agg in ["mean", "sum"]
    if f"highly_variable_{n_top_genes}" not in adata.var_keys():
        raise ("Please use sc.pp.highly_variable_genes first if you want to use hvg.")
    
    df = adata[:, adata.var[f"highly_variable_{n_top_genes}"]].to_df(layer=layer)
    df.index = df.index.map(adata.obs[label_key].to_dict())
    df.index.name = label_key
    expr = df.groupby(label_key).apply(agg)
    adata.uns[f"{label_key}_expr_{n_top_genes}"] = expr
    adata.uns[f"{label_key}_expr_{n_top_genes}_params"] = {"n_top_genes": n_top_genes, "aggregation": agg}
    
def _reference_highly_variable_genes(
    adata: ad.AnnData,
    n_top_genes: int = 20000,
    *,
    layer: str | None = "counts",
    flavor: Literal["seurat_v3", "cell_ranger"] = "seurat_v3",
):
    res = sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, layer=layer, flavor=flavor, inplace=False)
    adata.var[f"highly_variable_{n_top_genes}"] = res["highly_variable"]
    
def preprocess_reference(
    adata: ad.AnnData,
    label_key: str,
    n_top_genes: int,
    save_folder: str | Path,
    *,
    layer: str | None = "counts",
) -> None:
    if f"highly_variable_{n_top_genes}" not in adata.var_keys():
        _reference_highly_variable_genes(adata, n_top_genes=n_top_genes, layer=layer)
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    print(f"calculating cell type expression for {label_key}...")
    _reference_calculate_cell_type_expression(adata, label_key, layer=layer, n_top_genes=n_top_genes)

    print(f"saving hv_genes and expr to {save_folder}...")
    joblib.dump(adata.var[f"highly_variable_{n_top_genes}"], save_folder.joinpath(f"hv_genes_{n_top_genes}.pkl"))
    joblib.dump(adata.uns[f"{label_key}_expr_{n_top_genes}"], save_folder.joinpath(f"{label_key}_expr_{n_top_genes}.pkl"))
