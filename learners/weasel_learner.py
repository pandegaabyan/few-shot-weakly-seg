from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor

from config.config_type import ConfigWeasel
from data.typings import FewSparseDataTuple
from learners.meta_learner import MetaLearner
from torchmeta.modules.module import MetaModule


class WeaselLearner(MetaLearner[ConfigWeasel], ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigWeasel)

        self.net = self.make_net()
        assert isinstance(self.net, MetaModule)

        self.automatic_optimization = False

    @abstractmethod
    def make_net(self) -> MetaModule:
        pass

    def forward(
        self, supp_image: Tensor, supp_mask: Tensor, qry_image: Tensor
    ) -> Tensor:
        s_images, s_masks, qry_images = self.split_tensors(
            [supp_image, supp_mask, qry_image]
        )

        with torch.enable_grad():
            for i, (s_image, s_mask) in enumerate(zip(s_images, s_masks)):
                s_pred = self.net(s_image)
                new_loss = self.loss(s_pred, s_mask) * s_image.size(0)
                if i == 0:
                    supp_loss = new_loss
                else:
                    supp_loss += new_loss

        with self.profile("update_parameters"):
            params = self.update_parameters(supp_loss)

        qry_pred_list = []
        for q_image in qry_images:
            q_pred = self.net(q_image, params=params)
            qry_pred_list.append(q_pred)
        qry_pred = torch.vstack(qry_pred_list)

        return qry_pred

    def training_process(
        self, batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor]:
        support, query, _ = batch
        with self.profile("forward"):
            pred = self.forward(support.images, support.masks, query.images)
        loss = self.loss(pred, query.masks)

        self.net.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self.manual_optimizer_step()
        if self.trainer.is_last_batch:
            self.manual_scheduler_step()

        return pred, loss

    def evaluation_process(
        self, type: Literal["VL", "TS"], batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        support, query, _ = batch
        s_images, s_masks, qry_images = self.split_tensors(
            [support.images, support.masks, query.images]
        )

        net_state_dict = deepcopy(self.net.state_dict())
        opt_state_dicts = deepcopy(self.optimizer_state_dicts())

        tune_epochs = self.config["weasel"]["tune_epochs"]
        tune_val_freq = self.config["weasel"]["tune_val_freq"]
        for ep in range(tune_epochs):
            if type == "VL":
                if self.trainer.sanity_checking:
                    progress_task = "sanity"
                else:
                    progress_task = "val"
            else:
                progress_task = "test"
            self.update_progress_bar_fields(
                progress_task, tune_epoch=f"{ep}/{tune_epochs-1}"
            )

            with self.profile("tune_process"):
                self.tune_process(s_images, s_masks)

            last_epoch = ep == tune_epochs - 1
            if (not last_epoch) and (
                tune_val_freq is None or ((ep + 1) % tune_val_freq != 0)
            ):
                continue

            with torch.inference_mode():
                self.net.eval()
                qry_pred_list = []
                for q_image in qry_images:
                    with self.profile("prediction"):
                        q_pred = self.net(q_image)
                    qry_pred_list.append(q_pred)
                    self.profile_post_process(q_pred)
                qry_pred = torch.vstack(qry_pred_list)
                qry_loss = self.loss(qry_pred, query.masks)
                if last_epoch:
                    qry_score = self.metric(qry_pred, query.masks)
                else:
                    qry_score = self.metric.measure(qry_pred, query.masks)

            if self.trainer.sanity_checking or tune_val_freq is None:
                continue

            self.log_to_table_tuning_metrics(type, batch_idx, ep, qry_loss, qry_score)

        self.net.load_state_dict(net_state_dict)
        self.load_optimizer_state_dicts(opt_state_dicts)

        return qry_pred, qry_loss, qry_score

    def log_to_table_tuning_metrics(
        self,
        type: Literal["VL", "TS"],
        batch_idx: int,
        tune_epoch: int,
        loss: Tensor,
        score: dict[str, Tensor],
    ):
        if not self.config["log"].get("table"):
            return
        score_tup = self.metric.prepare_for_log(score)
        self.log_table(
            [
                ("type", type),
                ("epoch", 0 if type == "TS" else self.current_epoch),
                ("batch", batch_idx),
                ("tune_epoch", tune_epoch),
                ("loss", loss.item()),
            ]
            + score_tup,
            "tuning_metrics",
        )

    def update_parameters(self, loss: Tensor) -> OrderedDict[str, Tensor]:
        if self.training:
            create_graph = not self.config["weasel"]["first_order"]
        else:
            create_graph = False

        grads = torch.autograd.grad(
            loss,
            list(self.net.parameters()),
            create_graph=create_graph,
        )

        params = OrderedDict()
        for (name, param), grad in zip(self.net.meta_named_parameters(), grads):
            new_param = param - self.config["weasel"]["update_param_rate"] * grad
            params[name] = new_param

        return params

    def tune_process(self, images: tuple[Tensor, ...], masks: tuple[Tensor, ...]):
        with torch.enable_grad():
            self.net.train()
            if self.config["weasel"]["tune_multi_step"]:
                for image, mask in zip(images, masks):
                    pred = self.net(image)
                    loss = self.loss(pred, mask)
                    self.net.zero_grad(set_to_none=True)
                    self.manual_backward(loss)
                    self.manual_optimizer_step()
            else:
                for i, (image, mask) in enumerate(zip(images, masks)):
                    pred = self.net(image)
                    new_loss = self.loss(pred, mask) * image.size(0)
                    if i == 0:
                        loss = new_loss
                    else:
                        loss += new_loss
                self.net.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                self.manual_optimizer_step()

    def profile_post_process(self, q_pred: Tensor):
        if (
            self.config["learn"].get("profiler") is None
            or self.trainer.state.fn != "test"
        ):
            return
        with self.profile("post_process"):
            is_binary = self.config["data"]["num_classes"] == 2
            if is_binary:
                tensor = (q_pred > 0).long().squeeze(1)
            else:
                tensor = q_pred.argmax(dim=1)
            _ = tensor.cpu().numpy()
