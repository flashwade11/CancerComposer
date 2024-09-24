from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def _align_expr(
    bulk: ad.AnnData,
    hv_genes: pd.Series,
    n_top_genes: int,
    copy: bool = False,
) ->  ad.AnnData | None:
    expr = bulk.uns["expr"]
    if isinstance(expr, (np.ndarray, Sequence)):
        if all(isinstance(item, pd.DataFrame) for item in expr):
            bulk.uns[f"expr_{n_top_genes}"] = list(map(lambda df: df.loc[:, hv_genes].values, expr))
        elif all(isinstance(item, np.ndarray) for item in expr):
            bulk.uns[f"expr_{n_top_genes}"] = bulk.uns["expr"][:, :, hv_genes]
        else:
            raise ValueError("expr should be a Sequence of pd.DataFrame or np.ndarray")
    else:
        raise ValueError(f"expr should be a sequence of pd.DataFrame or np.ndarray, but got {type(expr)}")
    return bulk if copy else None

def _log1p_expr(
    bulk: ad.AnnData,
    n_top_genes: int,
    copy: bool = False,
):
    bulk.uns[f"expr_{n_top_genes}_log1p"] = list(map(np.log1p, bulk.uns[f"expr_{n_top_genes}"]))
    return bulk if copy else None

def _align_X(
    bulk: ad.AnnData,
    hv_genes: pd.Series,
    n_top_genes: int,
    copy: bool = False,
):
    bulk.obsm[f"X_{n_top_genes}"] = bulk.X[:, hv_genes].copy()
    return bulk if copy else None

def _mms_X_obsm(
    bulk: ad.AnnData,
    n_top_genes: int,
    copy: bool = False,
):
    bulk.obsm[f"X_{n_top_genes}_MMS"] = minmax_scale(bulk.obsm[f"X_{n_top_genes}"], axis=1)
    return bulk if copy else None


def align_highly_variable_genes(
    bulk: ad.AnnData,
    hv_genes: pd.Series,
    copy: bool = False,
):
    n_top_genes = int(hv_genes.sum())
    bulk.var[f"hv_genes_{n_top_genes}"] = hv_genes
    _align_X(bulk, hv_genes, n_top_genes)
    _mms_X_obsm(bulk, n_top_genes)
    _align_expr(bulk, hv_genes, n_top_genes)
    _log1p_expr(bulk, n_top_genes)
    return bulk if copy else None
    