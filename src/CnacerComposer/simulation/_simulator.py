# https://github.com/flashwade11/PySimBu

from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Literal

import anndata as ad
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump


def simulate_sample(
    adata: ad.AnnData,
    layer: str | None,
    label_key: str,
    sample_proportion: pd.Series,
    total_cells: int,
) -> dict[str, pd.DataFrame]:
    if not np.isclose(sample_proportion.sum(), 1.0):
        raise ValueError(f"simulation_vector must sum up to 1 in each sample, but got {sample_proportion.sum()}.")

    cell_type_counts = (sample_proportion * total_cells).round().astype(int).replace(0, 1)

    sampled_cells_indices = [
        np.random.choice(
            adata.obs.index[adata.obs[label_key] == cell_type],
            size=count,
            replace=True,
        )
        for cell_type, count in cell_type_counts.items()
    ]
    sampled_cell_indices = np.concatenate(sampled_cells_indices) 
    sampled_adata = adata[sampled_cell_indices]

    # simulate bulk
    sampled_df = sampled_adata.to_df(layer=layer)
    simulated_bulk_count = sampled_df.sum(axis=0).to_frame().T

    # simulate cell type expression
    sampled_ct_mapping = sampled_adata.obs[label_key].to_dict()
    sampled_df.index = sampled_df.index.map(sampled_ct_mapping)
    sampled_df.index.name = label_key
    simulated_ct_expression = sampled_df.groupby(label_key).sum()

    return {"bulk": simulated_bulk_count, "cell_type_expression": simulated_ct_expression}


def simulate_prop(
    adata: ad.AnnData,
    label_key:str = "cell_type",
    scenario: Literal["even", "random", "weighted", "custom"] = "random",
    n_samples: int = 100,
    weighted_cell_type: str | None = None,
    weighted_amount: float | None = None,
    custom_scenario_dataframe: pd.DataFrame | None = None,
    balance_even_mirror_scenario: float = 0.01,
) -> tuple[pd.DataFrame, str]:
    cell_types = adata.obs[label_key].unique().tolist()
    n_cell_types = len(cell_types)
    simulation_vector_list = []
    if scenario == "even":
        for _ in range(n_samples):
            vector = np.round(
                np.random.normal(
                    loc=1.0 / n_cell_types,
                    scale=balance_even_mirror_scenario,
                    size=n_cell_types,
                ),
                3,
            )
            vector = vector / np.sum(vector)
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=cell_types)
    elif scenario == "random":
        for _ in range(n_samples):
            vector = np.round(np.random.uniform(0, 1, size=n_cell_types), 3)
            vector = vector / np.sum(vector)
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=cell_types)
    elif scenario == "weighted":
        if weighted_cell_type is None or weighted_amount is None:
            raise ValueError("weighted_cell_type and weighted_amount must be provided for weighted scenario")
        if weighted_amount > 0.99 or weighted_amount < 0:
            raise ValueError("weighted_amount must be between 0 and 0.99")
        if weighted_cell_type not in cell_types:
            raise ValueError(f"weighted_cell_type must be one of {cell_types}")
        random_cell_types = cell_types.copy()
        random_cell_types.remove(weighted_cell_type)
        random_cell_types.insert(0, weighted_cell_type)
        for _ in range(n_samples):
            noise = np.random.uniform(-0.01, 0.01, size=1)
            vector = np.round(np.random.uniform(0, 1, size=n_cell_types - 1), 3)
            vector = (1 - weighted_amount - noise) * vector / np.sum(vector)
            vector = np.insert(vector, 0, weighted_amount + noise)
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=random_cell_types)
    elif scenario == "custom":
        if custom_scenario_dataframe is None:
            raise ValueError("custom_scenario_dataframe must be provided for custom scenario")
        if custom_scenario_dataframe.shape[0] != n_samples:
            raise ValueError("custom_scenario_dataframe must have the same number of rows as n_samples")
        if not all(custom_scenario_dataframe.columns.isin(cell_types)):
            raise ValueError("Could not find all cell-types from scenario data in annotation.")
        simulation_vector = custom_scenario_dataframe
    else:
        raise ValueError("Scenario must be either 'even', 'random', 'weighted', or 'custom'")

    simulation_vector.index = simulation_vector.index.astype(str)
    simulation_vector = simulation_vector.reindex(columns=sorted(simulation_vector.columns))
    return simulation_vector


def simulate_bulk(
    adata: ad.AnnData,
    layer: str | None = None,
    label_key: str = "cell_type",
    scenario: Literal["even", "random", "weighted", "custom"] = "random",
    n_samples: int = 100,
    n_cells: int | Sequence[int] = 1000,
    weighted_cell_type: str | None = None,
    weighted_amount: float | None = None,
    custom_scenario_dataframe: pd.DataFrame | None = None,
    balance_even_mirror_scenario: float = 0.01,
    n_jobs: int = -1,
    verbose: int = 1,
) -> ad.AnnData:
    if label_key not in adata.obs.columns:
        raise ValueError(f"There is no column `{label_key}` in the adata.obs dataframe, please check your label_key.")
    
    if layer not in adata.layers.keys():
        raise ValueError(f"There is no layer `{layer}` in the adata.layers, please check your layer.")
    
    if isinstance(n_cells, int):
        n_cells = [n_cells] * n_samples
    elif isinstance(n_cells, (Sequence, np.ndarray)):
        if len(n_cells) != n_samples:
            raise ValueError(f"n_cells must be a list of integers with the same length as n_samples = {n_samples}, but got {len(n_cells)}.")
    else:
        raise ValueError("n_cells must be either an integer or a list of integers with the same length as n_samples.")
    
    simulation_vector = simulate_prop(
        adata,
        label_key,
        scenario,
        n_samples,
        weighted_cell_type,
        weighted_amount,
        custom_scenario_dataframe,
        balance_even_mirror_scenario,
    )

    simulation_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(simulate_sample)(adata, layer, label_key, sample_vector, total_cells=n_c)
        for n_c, (_, sample_vector) in zip(n_cells, simulation_vector.iterrows())
    )

    bulk_counts = pd.concat([res["bulk"] for res in simulation_results], ignore_index=True) # type: ignore
    bulk_counts.index = bulk_counts.index.astype(str)

    cell_type_expression = [res["cell_type_expression"] for res in simulation_results] # type: ignore
    simulation = ad.AnnData(
        X=bulk_counts,
        obsm=dict(prop=simulation_vector), # type: ignore
        uns=dict(expr=cell_type_expression),
    )

    return simulation


def merge_simulation(simulation_list: list[ad.AnnData]) -> ad.AnnData:
    simulation = ad.concat(simulation_list)
    simulation.obs.index = pd.Index(range(simulation.shape[0]), dtype=str)
    simulation.uns["expr"] = list(chain.from_iterable([s.uns["expr"] for s in simulation_list]))
    return simulation


def save_simulation(simulation: ad.AnnData, output_file: str | Path):
    dump(simulation, output_file)
