import os

from pytorch_lightning import Trainer

from config.config_type import ConfigUnion
from config.constants import FILENAMES
from runners.callbacks import make_callbacks


def make_trainer(config: ConfigUnion, **kwargs) -> Trainer:
    callbacks = make_callbacks(
        config["callbacks"],
        os.path.join(
            FILENAMES["checkpoint_folder"],
            config["learn"]["exp_name"],
            config["learn"]["run_name"],
        ),
        config["learn"].get("val_freq", 1),
    )
    default_kwargs: dict = {"num_sanity_val_steps": 0}
    return Trainer(
        max_epochs=config["learn"]["num_epochs"],
        check_val_every_n_epoch=config["learn"].get("val_freq", 1),
        callbacks=callbacks,
        logger=False,
        inference_mode=not config["learn"].get("manual_optim", False),
        **(default_kwargs | kwargs),
    )
