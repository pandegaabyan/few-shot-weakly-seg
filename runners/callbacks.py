from typing import Literal

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.progress import TextColumn

from config.config_type import CallbacksConfig

ProgressBarTaskType = Literal["train", "val", "test", "predict", "sanity"]


class CustomRichProgressBar(RichProgressBar):
    def on_sanity_check_end(self, trainer, pl_module):
        self.refresh()

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self.is_enabled
            and self.val_progress_bar_id is not None
            and trainer.state.fn == "fit"
        ):
            self.refresh()

    def configure_columns(self, trainer):
        return super().configure_columns(trainer) + [
            TextColumn(
                "{task.fields}",
                style=self.theme.description,
            )
        ]

    def _add_task(self, total_batches, description, visible=True):
        return super()._add_task(total_batches, description, True)

    def update_fields(self, task: ProgressBarTaskType, **kwargs):
        if self.progress is None:
            return

        if task == "train":
            task_id = self.train_progress_bar_id
        elif task == "val":
            task_id = self.val_progress_bar_id
        elif task == "test":
            task_id = self.test_progress_bar_id
        elif task == "predict":
            task_id = self.predict_progress_bar_id
        elif task == "sanity":
            task_id = self.val_sanity_progress_bar_id
        else:
            return

        if task_id is None:
            return

        self.progress.update(task_id, **kwargs)
        self.refresh()


def make_callbacks(
    config: CallbacksConfig, ckpt_path: str, ckpt_every_n_epochs: int = 1
) -> list[Callback]:
    progress_callback = CustomRichProgressBar(
        leave=config.get("progress_leave", False),
        theme=RichProgressBarTheme(
            description="#6206E0",
            batch_progress="#6206E0",
            time="#6206E0",
            processing_speed="#6206E0",
            metrics="#6206E0",
        ),
    )

    monitor = config.get("monitor", None)
    monitor_mode = config.get("monitor_mode", "min")
    ckpt_filename = ("{epoch} {" + monitor + ":.2f}") if monitor else ("{epoch}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=ckpt_filename,
        monitor=monitor,
        mode=monitor_mode,
        save_last=config.get("ckpt_last", True),
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
