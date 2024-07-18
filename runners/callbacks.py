from typing import Literal

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.progress import TextColumn

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


custom_rich_progress_bar_theme = RichProgressBarTheme(
    description="#6206E0",
    batch_progress="#6206E0",
    time="#6206E0",
    processing_speed="#6206E0",
    metrics="#6206E0",
)
