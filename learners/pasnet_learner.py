from typing import Literal, Type

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torchvision.transforms import v2

from config.config_type import ConfigPASNet
from data.few_sparse_dataset import FewSparseDataset
from data.typings import DatasetModes, FewSparseDatasetKwargs, FewSparseDataTuple
from learners.meta_learner import MetaLearner
from learners.models import make_segmentation_model
from learners.optimizers import make_optimizer_adam, make_scheduler_step


class PASNetLearner(MetaLearner[ConfigPASNet]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigPASNet)

        self.net = self.make_net()

        self.support_embedding: Tensor | None = None
        self.transformed_support_embedding: Tensor | None = None
        self.query_embedding: Tensor | None = None

        self.num_classes = self.config["data"]["num_classes"]
        self.par_weight = self.config["pasnet"]["par_weight"]
        self.consistency_weight = self.config["pasnet"]["consistency_weight"]
        self.prototype_metric_func = self.config["pasnet"]["prototype_metric_func"]
        self.consistency_metric_func = self.config["pasnet"]["consistency_metric_func"]
        self.high_confidence_threshold = self.config["pasnet"][
            "high_confidence_threshold"
        ]

        self.style_transforms = v2.Compose(
            [
                v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2,
                ),
                v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
                v2.RandomApply(
                    nn.ModuleList([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))]),
                    p=0.2,
                ),
            ]
        )

    def make_dataset(
        self,
        mode: DatasetModes,
        datasets: list[tuple[Type[FewSparseDataset], FewSparseDatasetKwargs]],
    ) -> list[FewSparseDataset]:
        for _, kwargs in datasets:
            kwargs["scaling"] = None
        return super().make_dataset(mode, datasets)

    def make_net(self) -> nn.Module:
        return make_segmentation_model(
            self.config["model"],
            self.config["data"]["num_channels"],
            self.config["pasnet"]["embedding_size"],
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        adam_optimizer = make_optimizer_adam(self.config["optimizer"], self.net)
        step_scheduler = make_scheduler_step(adam_optimizer, self.config["scheduler"])
        return [adam_optimizer], [step_scheduler]

    def forward(
        self, supp_image: Tensor, supp_mask: Tensor, qry_image: Tensor
    ) -> Tensor:
        # tup [B C H W], tup [B H W], tup [B C H W]
        s_images, s_masks, q_images = self.split_tensors(
            [supp_image, supp_mask, qry_image]
        )

        with self.profile("get_transformed_support_embedding"):
            ts_emb_linear_list = []
            for s_image in s_images:
                ts_image = self.style_transforms(s_image)
                ts_image = self.scale_image(ts_image)
                ts_emb: Tensor = self.net(ts_image)  # [B E H W]
                ts_emb_linear = self.linearize_embeddings(ts_emb)  # [B H*W E]
                ts_emb_linear_list.append(ts_emb_linear)
            self.transformed_support_embedding = torch.vstack(
                ts_emb_linear_list
            )  # [S H*W E]

        with self.profile("get_prototypes"):
            s_emb_linear_list, s_mask_linear_list = [], []
            for s_image, s_mask in zip(s_images, s_masks):
                s_image = self.scale_image(s_image)
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
                q_image = self.scale_image(q_image)
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

        consistency_loss = self.calc_consistency_loss()

        total_loss = (
            loss
            + self.par_weight * par_loss
            + self.consistency_weight * consistency_loss
        )

        return (pred, total_loss)

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

        consistency_loss = self.calc_consistency_loss()

        total_loss = (
            loss
            + self.par_weight * par_loss
            + self.consistency_weight * consistency_loss
        )

        return (pred, total_loss, score)

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

        if self.prototype_metric_func == "euclidean":
            return -torch.sum(
                (prototypes.unsqueeze(0).unsqueeze(2) - embeddings.unsqueeze(1)) ** 2,
                dim=-1,
            )
        elif self.prototype_metric_func == "cosine":
            # Normalize the prototypes and embeddings along the E dimension
            prototypes_normalized = F.normalize(prototypes, dim=1)
            embeddings_normalized = F.normalize(embeddings, dim=2)

            # Calculate the dot product using matmul
            # Multiply [B, H*W, E] by [E, C] -> [B, H*W, C]
            similarities = torch.matmul(embeddings_normalized, prototypes_normalized.T)

            # [B, H*W, C] -> [B, C, H*W]
            return similarities.permute(0, 2, 1)
        else:
            raise ValueError(
                f"Unsupported prototype metric function: {self.prototype_metric_func}"
            )

    def get_support_predictions(self, qry_pred: Tensor) -> Tensor:
        assert self.query_embedding is not None and self.support_embedding is not None

        qry_pred_shape = qry_pred.shape  # [Q C H W] or [Q 1 H W] if binary
        if qry_pred_shape[1] == 1:
            qry_probs = torch.sigmoid(qry_pred).squeeze(1)  # [Q H W]
            low_confidence_mask = qry_probs < self.high_confidence_threshold  # [Q H W]
            qry_pred_label = (qry_probs > 0.5).type(torch.int64)  # [Q H W]
        else:
            qry_probs = F.softmax(qry_pred, dim=1)  # [Q C H W]
            max_probs, qry_pred_label = torch.max(qry_probs, dim=1)  # [Q H W]
            low_confidence_mask = max_probs < self.high_confidence_threshold  # [Q H W]
        qry_pred_label[low_confidence_mask] = -1

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

    def calc_consistency_loss(self) -> Tensor:
        emb1, emb2 = self.transformed_support_embedding, self.support_embedding
        assert emb1 is not None and emb2 is not None

        if self.consistency_metric_func == "euclidean":
            return torch.mean(torch.sum((emb1 - emb2) ** 2, dim=-1))
        elif self.consistency_metric_func == "cosine":
            emb1_normalized = F.normalize(emb1, dim=-1)
            emb2_normalized = F.normalize(emb2, dim=-1)
            return torch.mean(1 - torch.sum(emb1_normalized * emb2_normalized, dim=-1))
        else:
            raise ValueError(
                f"Unsupported consistency metric function: {self.consistency_metric_func}"
            )

    def scale_image(self, image: Tensor) -> Tensor:
        return image.to(torch.float32) / 255.0

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
