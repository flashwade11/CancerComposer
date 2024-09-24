from pathlib import Path

import anndata as ad
import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from ._lightning_module import DataModule, ModelModule
        
        
class DeconvInterface:
    def __init__(
        self,
        logger: WandbLogger | None = None,
    ) -> None:
        self.run_setup = False
        self.logger = logger
        self._init_callbacks()

    def _init_callbacks(self,):
        self.callbacks = [
            RichModelSummary(max_depth=2),
            RichProgressBar(),
        ]
        
        if self.logger is not None:
            self.callbacks.append(LearningRateMonitor(logging_interval="step"))
            self.callbacks.append(ModelCheckpoint(
                dirpath=Path(self.logger.experiment.dir).parent.joinpath("checkpoints"),
                filename="{epoch}-{val_loss_total:.3f}-{val_PearsonCorrCoef:.2f}",
                monitor="val_PearsonCorrCoef",
                mode="max",
            ))
    
    def extend_callbacks(self, callbacks: list[Callback]) -> None:
        self.callbacks.extend(callbacks)
            
    def _check_setup(self) -> bool:
        if not self.run_setup:
            raise ValueError("Model and trainer have not been set up. Please call setup() first.")
        return True
    
    def setup(
        self,
        model_params: dict,
        reference_cell_type_expr: pd.DataFrame | None,
        trainer_params: dict | None,
        froce_resetup: bool = False,
    ):
        if self.run_setup and not froce_resetup:
            print("Model and trainer have already been set up. If you want to re-setup the model, set froce_resetup=True.")
            return
        
        if trainer_params is None:
            trainer_params = {
                "max_epochs": 3000
            }
        
        if self.logger:
            self.logger.experiment.config.update({
                "model": model_params,
                "trainer": trainer_params
            })

        self.model_module = ModelModule(
            model_params=model_params,
            reference_cell_type_expr=reference_cell_type_expr,
        )

        self.trainer = pl.Trainer(
            **{
                **trainer_params,
                **dict(callbacks=self.callbacks, logger=self.logger)
            }
        )

        self.run_setup = True

    def fit(
        self,
        bulk_dict: dict[str, ad.AnnData] | None = None,
        data_folder: str | Path | None = None,
        file_name_dict: dict[str, list[str]] | None = None,
        n_top_genes: int = 5000,
        batch_size: int = 512,
        # lr: float = 1e-3,
        *args,
        **kwargs
    ):
        assert (bulk_dict is not None) or (data_folder is not None and file_name_dict is not None), \
            ValueError("Either bulk_dict or data_folder and file_name_dict should be provided.")
        self._check_setup()

        data_module = DataModule(
            bulk_dict=bulk_dict,
            data_folder=data_folder,
            file_name_dict=file_name_dict,
            n_top_genes=n_top_genes,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

        self.trainer.fit(model=self.model_module, datamodule=data_module)
    
    def test(
        self,
        data_folder: str | Path,
        file_name_list: list[str],
        layer: str,
        *,
        batch_size: int = 500,
    ):
        # self._check_setup()
        
        # data_module = DataModule(
        #     data_folder=data_folder,
        #     file_name_list=file_name_list,
        #     layer=layer,
        #     batch_size=batch_size,
        # )
        # return self.trainer.test(model=self.model_module, datamodule=data_module)
        raise NotImplementedError("Test function is not implemented yet.")
        
    def predict(
        self,
        bulk_dict: dict[str, ad.AnnData] | None = None,
        data_folder: str | Path | None = None,
        file_name_dict: dict[str, list[str]] | None = None,
        layer: str | None = None,
        *,
        batch_size: int = 500,
    ):
        assert bulk_dict is not None or (data_folder is not None and file_name_dict is not None)
        self._check_setup()
        
        if bulk_dict:
            data_module = DataModule(
                bulk_dict=bulk_dict,
                layer=layer,
                batch_size=batch_size,
            )
        else:
            data_module = DataModule.load_from_files(
                data_folder=data_folder,
                file_name_dict=file_name_dict,
                layer=layer,
            )
        
        return self.trainer.predict(model=self.model_module, datamodule=data_module)
        
    def load(self, ckpt_file: str | Path, map_location: str | None = None):
        # self._check_setup()
        
        # self.model_module = ModelModule.load_from_checkpoint(
        #     checkpoint_path=ckpt_file,
        #     map_location=map_location,
        # )
        
        raise NotImplementedError