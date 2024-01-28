import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from config.config_type import AllConfig
from data.dataset_loaders import DatasetLoaderItem, DatasetLoaderParamReduced
from data.types import TensorDataItem
from learners.losses import CustomLoss
from learners.meta_learner import MetaLearner
from learners.types import CalcMetrics, NeuralNetworks, Optimizer, Scheduler


class GuidedNetsLearner(MetaLearner):
    def __init__(
        self,
        net: NeuralNetworks,
        config: AllConfig,
        meta_params: list[DatasetLoaderParamReduced],
        tune_param: DatasetLoaderParamReduced,
        calc_metrics: CalcMetrics | None = None,
        calc_loss: CustomLoss | None = None,
        optimizer: Optimizer | None = None,
        scheduler: Scheduler | None = None,
    ):
        super().__init__(
            net,
            config,
            meta_params,
            tune_param,
            calc_metrics,
            calc_loss,
            optimizer,
            scheduler,
        )
        assert isinstance(net, dict), "net should be dict of nn.Module"
        self.net = net

        if self.net.get("image") is None or not isinstance(
            self.net["image"], nn.Module
        ):
            raise ValueError('net should have "image" nn.Module')
        if self.net.get("mask") is None:
            self.set_default_mask_net()
        if self.net.get("merge") is None:
            self.set_default_merge_net()
        if self.net.get("head") is None:
            self.set_default_head_net()

    def set_used_config(self) -> list[str]:
        return super().set_used_config() + ["guidednets"]

    def meta_train_test_step(
        self, train_data: TensorDataItem, test_data: TensorDataItem
    ) -> float:
        x_tr, _, y_tr, _ = train_data
        x_ts, y_ts, _, _ = test_data
        if self.config["learn"]["use_gpu"]:
            x_tr = x_tr.cuda()
            y_tr = y_tr.cuda()
            x_ts = x_ts.cuda()
            y_ts = y_ts.cuda()

        for net in self.net.values():
            net.zero_grad()

        embeddings = self.get_embeddings(x_tr, y_tr)
        prototypes = self.get_prototypes(embeddings, list(x_ts.shape))
        p_ts = self.get_predictions(x_ts, prototypes)

        if self.calc_loss.loss_type == "mce":
            self.calc_loss.set_mce_weights_from_target(
                y_tr,
                self.config["data"]["num_classes"],
                use_gpu=self.config["learn"]["use_gpu"],
            )
        loss = self.calc_loss(p_ts, y_ts)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def tune_train_test_process(
        self, epoch: int, tune_loader: DatasetLoaderItem
    ) -> tuple[list[NDArray], list[NDArray], list[str]]:
        labels, preds, names = [], [], []

        with torch.no_grad():
            for net in self.net.values():
                net.eval()
                net.zero_grad()

            embed_tr_list = []
            x_tr: Tensor = Tensor()
            y_tr: Tensor = Tensor()
            for data in tune_loader["train"]:
                x_tr, _, y_tr, _ = data
                if self.config["learn"]["use_gpu"]:
                    x_tr = x_tr.cuda()
                    y_tr = y_tr.cuda()

                embed_tr = self.get_embeddings(x_tr, y_tr)
                embed_tr_list.append(embed_tr)

            if len(embed_tr_list) == 0:
                return labels, preds, names
            all_embed_tr = torch.vstack(embed_tr_list)
            prototypes = self.get_prototypes(
                all_embed_tr, [1, 1, x_tr.shape[2], x_tr.shape[3]]
            )

            for i, data in enumerate(tune_loader["test"]):
                x_ts, y_ts, _, img_name = data

                if self.config["learn"]["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                p_ts = self.get_predictions(x_ts, prototypes)

                labels.append(y_ts.cpu().numpy().squeeze())
                preds.append(p_ts.argmax(1).cpu().numpy().squeeze())
                names.append(img_name[0])

        return labels, preds, names

    @staticmethod
    def one_hot_masks(mask: Tensor, num_classes: int) -> list[Tensor]:
        c_masks = []
        for c in range(num_classes):
            c_mask = torch.zeros_like(mask).float()
            c_mask[mask == c] = 1
            c_mask.unsqueeze_(1)
            c_masks.append(c_mask)

        return c_masks

    def get_embeddings(self, image: Tensor, mask: Tensor) -> Tensor:
        # TODO: alternative mask handling is to use one-hot single mask Tensor as input to network
        image_embedding = self.net["image"](image)
        c_masks = self.one_hot_masks(mask, self.config["data"]["num_classes"])

        mask_embeddings = [self.net["mask"](c_mask) for c_mask in c_masks]
        combined_embeddings = torch.clone(image_embedding)
        for mask_embedding in mask_embeddings:
            combined_embeddings *= mask_embedding

        merged_embeddings = self.net["merge"](combined_embeddings)

        return merged_embeddings

    @staticmethod
    def get_prototypes(embeddings: Tensor, ref_shape: list[int]) -> Tensor:
        proto = torch.mean(embeddings, dim=0, keepdim=True)
        proto = torch.tile(proto, (ref_shape[0], 1, ref_shape[2], ref_shape[3]))
        return proto

    def get_predictions(self, image: Tensor, prototypes: Tensor) -> Tensor:
        image_embedding = self.net["image"](image)
        mask_embedding = torch.cat([prototypes, image_embedding], dim=1)
        mask = self.net["head"](mask_embedding)
        return mask

    def set_default_mask_net(self):
        self.net["mask"] = nn.Identity()

    def set_default_merge_net(self):
        self.net["merge"] = nn.AdaptiveAvgPool2d((1, 1)).cuda()

    def set_default_head_net(self):
        embedding_size = self.config["guidednets"]["embedding_size"]  # type: ignore
        self.net["head"] = nn.Sequential(
            nn.Conv2d(embedding_size * 2, embedding_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                embedding_size, self.config["data"]["num_classes"], kernel_size=1
            ),
        ).cuda()
        nn.init.ones_(self.net["head"][0].weight)
        nn.init.ones_(self.net["head"][-1].weight)
