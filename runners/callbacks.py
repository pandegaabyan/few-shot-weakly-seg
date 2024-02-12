from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from config.config_type import CallbacksConfig


class CustomRichProgressBar(RichProgressBar):
    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ...

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if (
            self.is_enabled
            and self.val_progress_bar_id is not None
            and trainer.state.fn == "fit"
        ):
            self.refresh()


def make_callbacks(
    config: CallbacksConfig, ckpt_path: str, ckpt_every_n_epochs: int = 1
) -> list[Callback]:
    progress_callback = CustomRichProgressBar(leave=config.get("progress_leave", False))
    ckpt_monitor = config.get("ckpt_monitor", None)
    ckpt_filename = (
        ("{epoch} {" + ckpt_monitor + ":.2f}") if ckpt_monitor else ("{epoch}")
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        save_last=True,
        save_top_k=config.get("ckpt_top_k", 0),
        mode=config.get("ckpt_mode", "min"),
        every_n_epochs=ckpt_every_n_epochs,
        # save_on_train_epoch_end=True,
    )
    return [progress_callback, checkpoint_callback]
