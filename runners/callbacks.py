from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

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

    monitor = config.get("monitor", None)
    monitor_mode = config.get("monitor_mode", "min")
    ckpt_filename = ("{epoch} {" + monitor + ":.2f}") if monitor else ("{epoch}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=ckpt_filename,
        monitor=monitor,
        mode=monitor_mode,
        save_last=True,
        save_top_k=config.get("ckpt_top_k", 0),
        every_n_epochs=ckpt_every_n_epochs,
    )

    early_stopping_callback = monitor and EarlyStopping(
        monitor=monitor,
        mode=monitor_mode,
        verbose=True,
        patience=config.get("stop_patience", 3),
        min_delta=config.get("stop_min_delta", 0.0),
        stopping_threshold=config.get("stop_threshold", None),
    )

    callbacks = [progress_callback, checkpoint_callback]
    if early_stopping_callback:
        callbacks.append(early_stopping_callback)

    return callbacks
