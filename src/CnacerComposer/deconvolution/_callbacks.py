from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning, Callback
from torch.optim.optimizer import Optimizer


class TaskSwitcher(BaseFinetuning):
    def __init__(
        self,
        switch_epoch: int = 2000,
    ) -> None:
        super().__init__()
        self._switch_epoch = switch_epoch
        
    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(pl_module.model.expression_decoder)
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.model.task = "prop"
        print("The First task is trainning the two encoder and proportion decoder")
    
    def finetune_function(self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer) -> None:
        if current_epoch == self._switch_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.expression_decoder,
                optimizer=optimizer,
                train_bn=True,
            )
            
            self.freeze(pl_module.model.bulk_encoder)
            self.freeze(pl_module.model.reference_encoder)
            
            pl_module.model.task = "expr"
            print(f"At {self._switch_epoch} epoch, switching to expression task")
