from pathlib import Path
from typing import Literal

import wandb
from lightning.pytorch.loggers import WandbLogger


def init_wandb_logger(
    project_name: str,
    run_name: str,
    save_dir: str | Path,
    *,
    log_model: Literal["all"] | bool = False,
    tags: list[str] | None = None,
) -> WandbLogger:
    assert isinstance(save_dir, (str, Path))
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.joinpath(run_name).mkdir(parents=True, exist_ok=True)    
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        dir=save_dir.joinpath(run_name),
    )
    
    return WandbLogger(
        experiment=run,
        save_dir=save_dir,
        log_model=log_model,
        tags=tags
    )
