from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor

from config.config_type import ConfigPANet
from data.typings import FewSparseDataTuple
from learners.meta_learner import MetaLearner
from torchmeta.modules.module import MetaModule


class PANetLearner(MetaLearner[ConfigPANet], ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigPANet)

        self.net = self.make_net()

        self.support_embedding: Tensor | None = None
        self.query_embedding: Tensor | None = None

        self.num_classes = self.config["data"]["num_classes"]
        self.par_weight = self.config["panet"]["par_weight"]

    @abstractmethod
    def make_net(self) -> MetaModule:
        pass

    def forward(
        self, supp_image: Tensor, supp_mask: Tensor, qry_image: Tensor
    ) -> Tensor:
        # tup [B C H W], tup [B H W], tup [B C H W]
        s_images, s_masks, q_images = self.split_tensors(
            [supp_image, supp_mask, qry_image]
        )

        with self.profile("get_prototypes"):
            s_emb_linear_list, s_mask_linear_list = [], []
            for s_image, s_mask in zip(s_images, s_masks):
                s_emb: Tensor = self.net(s_image)  # [B E H W]
                s_emb_linear = self.linearize_embeddings(s_emb)  # [B H*W E]
                s_mask_linear = s_mask.view(s_mask.size(0), -1)  # [B H*W]
                s_emb_linear_list.append(s_emb_linear)
                s_mask_linear_list.append(s_mask_linear)
            self.support_embedding = torch.vstack(s_emb_linear_list)  # [S H*W E]
            support_mask = torch.vstack(s_mask_linear_list)  # [S H*W]
            prototypes = self.get_prototypes(
                self.support_embedding, support_mask, self.num_classes
            )  # [C E]

        q_emb_linear_list, q_pred_list = [], []
        for q_image in q_images:
            with self.profile("prediction"):
                q_emb: Tensor = self.net(q_image)  # [B E H W]
                q_emb_linear = self.linearize_embeddings(q_emb)  # [B H*W E]
                q_emb_linear_list.append(q_emb_linear)
                q_pred_linear = self.get_predictions(
                    prototypes, q_emb_linear
                )  # [B C H*W]
                q_pred = q_pred_linear.view(
                    *q_pred_linear.shape[:-1], *q_image.shape[2:]
                )  # [B C H W]
            q_pred_list.append(q_pred)
            self.profile_post_process(q_pred)
        self.query_embedding = torch.vstack(q_emb_linear_list)  # [Q H*W E]
        qry_pred = torch.vstack(q_pred_list)  # [Q C H W]

        # if C == 2, convert to binary (C == 1)
        if self.num_classes == 2:
            qry_pred = qry_pred[..., 0:1, :, :] - qry_pred[..., 1:2, :, :]

        return qry_pred  # [Q C H W]

    def training_process(
        self, batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor]:
        support, query, _ = batch
        with self.profile("forward"):
            pred = self.forward(support.images, support.masks, query.images)
        loss = self.loss(pred, query.masks)

        with self.profile("get_support_predictions"):
            supp_pred = self.get_support_predictions(pred)
        par_loss = self.loss(supp_pred, support.masks)

        return pred, loss + self.par_weight * par_loss

    def evaluation_process(
        self, type: Literal["VL", "TS"], batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        support, query, _ = batch
        with self.profile("forward"):
            pred = self.forward(support.images, support.masks, query.images)
        loss = self.loss(pred, query.masks)
        score = self.metric(pred, query.masks)

        with self.profile("get_support_predictions"):
            supp_pred = self.get_support_predictions(pred)
        par_loss = self.loss(supp_pred, support.masks)

        return pred, loss + self.par_weight * par_loss, score

    def linearize_embeddings(self, embeddings: Tensor) -> Tensor:
        # [B E H W] -> [B H*W E]

        return embeddings.permute(0, 2, 3, 1).view(
            embeddings.size(0),
            embeddings.size(2) * embeddings.size(3),
            embeddings.size(1),
        )

    def get_num_samples(self, targets: Tensor, num_classes: int, dtype=None) -> Tensor:
        # [S H*W] -> [C]

        batch_size = targets.size(0)
        num_classes = self.config["data"]["num_classes"]

        with torch.inference_mode():
            num_samples = targets.new_zeros((num_classes,), dtype=dtype)

            for i in range(batch_size):
                trg_i = targets[i]
                for c in range(num_classes):
                    s = trg_i[trg_i == c].size(0)
                    num_samples[c] += s

        return num_samples

    def get_prototypes(
        self, embeddings: Tensor, targets: Tensor, num_classes: int
    ) -> Tensor:
        # [S H*W E], [S H*W] -> [C E]

        batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

        num_samples = self.get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples = num_samples.unsqueeze(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = embeddings.new_zeros((num_classes, embedding_size))

        for i in range(batch_size):
            trg_i = targets[i]
            emb_i = embeddings[i]
            for c in range(num_classes):
                s = torch.sum(emb_i[trg_i == c], dim=0)
                prototypes[c] += s

        prototypes.div_(num_samples)

        return prototypes

    def get_predictions(self, prototypes: Tensor, embeddings: Tensor) -> Tensor:
        # [C E], [B H*W E] -> [B C H*W]

        def calc_squared_distances(proto: Tensor, embed: Tensor) -> Tensor:
            return torch.sum(
                (proto.unsqueeze(0).unsqueeze(2) - embed.unsqueeze(1)) ** 2, dim=-1
            )

        return -calc_squared_distances(prototypes, embeddings)

    def get_support_predictions(self, qry_pred: Tensor) -> Tensor:
        assert self.query_embedding is not None and self.support_embedding is not None

        qry_pred_shape = qry_pred.shape  # [Q C H W] or [Q 1 H W] if binary
        if qry_pred_shape[1] == 1:
            qry_pred_label = (qry_pred > 0.5).type(torch.int64).squeeze(1)  # [Q H W]
        else:
            qry_pred_label = qry_pred.argmax(dim=1).type(torch.int64)  # [Q H W]

        qry_pred_linear = qry_pred_label.view(
            qry_pred_shape[0], qry_pred_shape[2] * qry_pred_shape[3]
        )  # [Q H*W]

        qry_prototypes = self.get_prototypes(
            self.query_embedding, qry_pred_linear, self.num_classes
        )  # [C E]

        supp_pred_linear = self.get_predictions(
            qry_prototypes, self.support_embedding
        )  # [S C H*W]
        supp_pred = supp_pred_linear.view(
            *supp_pred_linear.shape[:-1], *qry_pred_shape[2:]
        )  # [S C H W]

        # if C == 2, convert to binary prediction
        if self.num_classes == 2:
            supp_pred = supp_pred[..., 0:1, :, :] - supp_pred[..., 1:2, :, :]

        return supp_pred  # [S 1 H W] if binary else [S C H W]

    def profile_post_process(self, q_pred: Tensor):
        # [B C H W] -> [B H W]

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
