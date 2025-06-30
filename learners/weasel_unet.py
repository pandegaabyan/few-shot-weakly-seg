from pytorch_lightning.utilities.types import OptimizerLRScheduler

from learners.optimizers import make_optimizer_adam, make_scheduler_step
from learners.weasel_learner import WeaselLearner
from models.unet import UNet
from torchmeta.modules.module import MetaModule


class WeaselUnet(WeaselLearner):
    def make_net(self) -> MetaModule:
        num_classes = self.config["data"]["num_classes"]
        output_channels = num_classes if num_classes != 2 else 1
        return UNet(self.config["data"]["num_channels"], output_channels)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        adam_optimizer = make_optimizer_adam(self.config["optimizer"], self.net)

        step_scheduler = make_scheduler_step(
            adam_optimizer,
            self.config["scheduler"],
        )

        return [adam_optimizer], [step_scheduler]
