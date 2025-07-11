from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor

from config.config_type import ConfigProtoSeg
from data.typings import FewSparseDataTuple
from learners.meta_learner import MetaLearner
from torchmeta.modules.module import MetaModule


class ProtosegLearner(MetaLearner[ConfigProtoSeg], ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigProtoSeg)

        self.net = self.make_net()

        self.multi_pred = self.config["protoseg"]["multi_pred"]

    @abstractmethod
    def make_net(self) -> MetaModule:
        pass

    def forward(
        self, supp_image: Tensor, supp_mask: Tensor, qry_image: Tensor
    ) -> Tensor:
        # tup [B C H W], tup [B H W], tup [B C H W]
        s_images, s_masks, qry_images = self.split_tensors(
            [supp_image, supp_mask, qry_image]
        )

        num_classes = self.config["data"]["num_classes"]

        with self.profile("get_prototypes"):
            s_emb_linear_list, s_mask_linear_list = [], []
            for s_image, s_mask in zip(s_images, s_masks):
                s_emb: Tensor = self.net(s_image)  # [B E H W]
                s_emb_linear = self.linearize_embeddings(s_emb)  # [B H*W E]
                s_mask_linear = s_mask.view(s_mask.size(0), -1)  # [B H*W]
                s_emb_linear_list.append(s_emb_linear)
                s_mask_linear_list.append(s_mask_linear)
            s_emb_linear = torch.vstack(s_emb_linear_list)  # [S H*W E]
            s_mask_linear = torch.vstack(s_mask_linear_list)  # [S H*W]
            prototypes = self.get_prototypes(
                s_emb_linear, s_mask_linear, num_classes
            )  # [S C E] if multi_pred else [C E]

        qry_pred_list = []
        for q_image in qry_images:
            with self.profile("prediction"):
                q_emb: Tensor = self.net(q_image)  # [B E H W]
                q_emb_linear = self.linearize_embeddings(q_emb)  # [B H*W E]
                q_pred_linear = self.get_predictions(
                    prototypes, q_emb_linear
                )  # [B S C H*W] if multi_pred else [B C H*W]
                q_pred = q_pred_linear.view(
                    *q_pred_linear.shape[:-1], *q_image.shape[2:]
                )  # [B S C H W] if multi_pred else [B C H W]
            qry_pred_list.append(q_pred)
            self.profile_post_process(q_pred)
        qry_pred = torch.vstack(qry_pred_list)

        # if C == 2, convert to binary (C == 1)
        if num_classes == 2:
            qry_pred = qry_pred[..., 0:1, :, :] - qry_pred[..., 1:2, :, :]

        return qry_pred  # [Q S C H W] if multi_pred else [Q C H W]

    def training_process(
        self, batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor]:
        support, query, _ = batch
        with self.profile("forward"):
            pred = self.forward(support.images, support.masks, query.images)
        if self.multi_pred:
            loss = self.loss(pred.mean(dim=1), query.masks)
        else:
            loss = self.loss(pred, query.masks)

        return pred, loss

    def evaluation_process(
        self, type: Literal["VL", "TS"], batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        support, query, _ = batch
        with self.profile("forward"):
            pred = self.forward(support.images, support.masks, query.images)
        if self.multi_pred:
            loss = self.loss(pred.mean(dim=1), query.masks)
            if self.config["data"]["num_classes"] == 2:
                pred_values = (pred > 0).squeeze(2).mode(dim=1).values
            else:
                pred_values = pred.argmax(dim=2).mode(dim=1).values
            score = self.metric(pred_values, query.masks)
        else:
            loss = self.loss(pred, query.masks)
            score = self.metric(pred, query.masks)

        return pred, loss, score

    def linearize_embeddings(self, embeddings: Tensor) -> Tensor:
        # [B E H W] -> [B H*W E]

        return embeddings.permute(0, 2, 3, 1).view(
            embeddings.size(0),
            embeddings.size(2) * embeddings.size(3),
            embeddings.size(1),
        )

    def get_num_samples(self, targets: Tensor, num_classes: int, dtype=None) -> Tensor:
        # [S H*W] -> [S C] if multi_pred else [C]

        batch_size = targets.size(0)
        num_classes = self.config["data"]["num_classes"]

        with torch.inference_mode():
            if self.multi_pred:
                num_samples_shape = (batch_size, num_classes)
            else:
                num_samples_shape = (num_classes,)
            num_samples = targets.new_zeros(num_samples_shape, dtype=dtype)

            for i in range(batch_size):
                trg_i = targets[i]
                for c in range(num_classes):
                    s = trg_i[trg_i == c].size(0)
                    if self.multi_pred:
                        num_samples[i, c] = s
                    else:
                        num_samples[c] += s

        return num_samples

    def get_prototypes(
        self, embeddings: Tensor, targets: Tensor, num_classes: int
    ) -> Tensor:
        # [S H*W E], [S H*W] -> [S C E] if multi_pred else [C E]

        batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

        num_samples = self.get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples = num_samples.unsqueeze(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        if self.multi_pred:
            prototypes_shape = (batch_size, num_classes, embedding_size)
        else:
            prototypes_shape = (num_classes, embedding_size)
        prototypes = embeddings.new_zeros(prototypes_shape)

        for i in range(batch_size):
            trg_i = targets[i]
            emb_i = embeddings[i]
            for c in range(num_classes):
                s = torch.sum(emb_i[trg_i == c], dim=0)
                if self.multi_pred:
                    prototypes[i, c] = s
                else:
                    prototypes[c] += s

        prototypes.div_(num_samples)

        return prototypes

    def get_predictions(self, prototypes: Tensor, embeddings: Tensor) -> Tensor:
        # multi_pred: [S C E], [B H*W E] -> [B S C H*W]
        # else: [C E], [B H*W E] -> [B C H*W]

        def calc_squared_distances(proto: Tensor, embed: Tensor) -> Tensor:
            return torch.sum(
                (proto.unsqueeze(0).unsqueeze(2) - embed.unsqueeze(1)) ** 2, dim=-1
            )

        if not self.multi_pred:
            return -calc_squared_distances(prototypes, embeddings)
        squared_distances_list = []
        for proto in prototypes:
            squared_distances_list.append(-calc_squared_distances(proto, embeddings))
        return torch.stack(squared_distances_list, dim=1)

    def profile_post_process(self, q_pred: Tensor):
        # multi_pred: [B S C H W] -> [B H W]
        # else: [B C H W] -> [B H W]

        if (
            self.config["learn"].get("profiler") is None
            or self.trainer.state.fn != "test"
        ):
            return

        with self.profile("post_process"):
            is_binary = self.config["data"]["num_classes"] == 2
            if self.multi_pred:
                if is_binary:
                    tensor = (q_pred > 0).long().squeeze(2).mode(dim=1).values
                else:
                    tensor = q_pred.argmax(dim=2).mode(dim=1).values
            else:
                if is_binary:
                    tensor = (q_pred > 0).long().squeeze(1)
                else:
                    tensor = q_pred.argmax(dim=1)
            _ = tensor.cpu().numpy()
