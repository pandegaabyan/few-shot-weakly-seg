from pytorch_lightning.utilities.types import OptimizerLRScheduler

from learners.optimizers import make_optimizer_adam, make_scheduler_step
from learners.protoseg_learner import ProtosegLearner
from models.u_net import UNet
from torchmeta.modules.module import MetaModule


class ProtosegUnet(ProtosegLearner):
    def make_net(self) -> MetaModule:
        return UNet(
            self.config["data"]["num_channels"], self.config["data"]["num_classes"]
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        adam_optimizer = make_optimizer_adam(self.config["optimizer"], self.net)

        step_scheduler = make_scheduler_step(
            adam_optimizer,
            self.config["scheduler"],
        )

        return [adam_optimizer], [step_scheduler]
