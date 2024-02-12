from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn, optim

from config.constants import DEFAULT_CONFIGS
from learners.simple_learner import SimpleLearner
from models.u_net import UNet


class SimpleUnet(SimpleLearner):
    def make_net(self) -> nn.Module:
        return UNet(
            self.config["data"]["num_channels"], self.config["data"]["num_classes"]
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        adam_optimizer = optim.Adam(
            [
                {
                    "params": [
                        param
                        for name, param in self.net.named_parameters()
                        if name[-4:] == "bias"
                    ],
                    "lr": self.config["optimizer"].get("lr_bias"),
                    "weight_decay": self.config["optimizer"].get("weight_decay_bias"),
                },
                {
                    "params": [
                        param
                        for name, param in self.net.named_parameters()
                        if name[-4:] != "bias"
                    ],
                    "lr": self.config["optimizer"].get("lr"),
                    "weight_decay": self.config["optimizer"].get("weight_decay"),
                },
            ],
            betas=self.config["optimizer"].get(
                "betas", DEFAULT_CONFIGS["optimizer_betas"]
            ),
        )

        step_scheduler = optim.lr_scheduler.StepLR(
            adam_optimizer,
            step_size=self.config["scheduler"].get(
                "step_size", DEFAULT_CONFIGS["scheduler_step_size"]
            ),
            gamma=self.config["scheduler"].get(
                "gamma", DEFAULT_CONFIGS["scheduler_gamma"]
            ),
        )

        return [adam_optimizer], [step_scheduler]
