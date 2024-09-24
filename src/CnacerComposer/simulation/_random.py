import gc
from pathlib import Path

import anndata as ad
import hdf5plugin
import numpy as np

from ._simulator import simulate_bulk


def simulate_bulk_at_random(
    adata: ad.AnnData,
    label_key: str,
    n_samples: int,
    *,
    train_val_test: tuple[int, int, int] = (8, 1, 1),
    layer: str = "counts",
    n_cpu: int = -1,
    save_folder: str | Path | None = None,
) -> None:
    assert label_key in adata.obs_keys(), f"{label_key} not in adata.obs_keys()"
    assert layer in adata.layers.keys(), f"{layer} not in adata.layers.keys()"

    assert save_folder is not None, "save_folder should be specified when save is True"
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
    
    n_train, n_val, n_test = train_val_test
    file_names = (
        [f"train_{i}" for i in range(1, n_train + 1)]
        + [f"val_{i}" for i in range(1, n_val + 1)]
        + [f"test_{i}" for i in range(1, n_test + 1)]
    )

    for file_name in file_names:
        n_cells = np.random.choice(range(1000, 5000), size=n_samples, replace=True)
        bulk = simulate_bulk(
            adata=adata, 
            label_key=label_key,
            layer=layer,
            n_samples=n_samples,
            n_cells=n_cells,
            n_jobs=n_cpu,
        )
        save_path = save_folder / f"{file_name}.h5ad"
        bulk.write_h5ad(save_path, compression=hdf5plugin.FILTERS["zstd"])
        
        del bulk
        gc.collect()