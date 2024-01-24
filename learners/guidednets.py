import os
from typing import Callable

import torch
from numpy.typing import NDArray
from torch import optim, nn, Tensor

from config.config_type import AllConfig
from config.constants import FILENAMES
from data.dataset_loaders import DatasetLoaderItem, DatasetLoaderParamReduced
from data.types import TensorDataItem
from learners.losses import CustomLoss
from learners.meta_learner import MetaLearner


class GuidedNetsLearner(MetaLearner):

    def __init__(self,
                 net_image: nn.Module,
                 net_mask: nn.Module | None,
                 net_merge: nn.Module,
                 net_head: nn.Module,
                 config: AllConfig,
                 meta_params: list[DatasetLoaderParamReduced],
                 tune_param: DatasetLoaderParamReduced,
                 calc_metrics: Callable[[list[NDArray], list[NDArray]], tuple[dict, str, str]] | None = None,
                 calc_loss: CustomLoss | None = None,
                 optimizer: optim.Optimizer | None = None,
                 scheduler: optim.lr_scheduler.LRScheduler | None = None):
        super().__init__(net_image, config, meta_params, tune_param,
                         calc_metrics, calc_loss, optimizer, scheduler)

        self.net_mask = net_mask
        self.net_merge = net_merge
        self.net_head = net_head

    def initialize_gpu_usage(self) -> tuple[float, int]:
        gpu_percent, gpu_total = super().initialize_gpu_usage()
        if self.config['learn']["use_gpu"]:
            self.net_merge.cuda()
            self.net_head.cuda()
            if self.net_mask is not None:
                self.net_mask.cuda()
        return gpu_percent, gpu_total

    def save_net_as_text(self):
        net_text = ''
        net_list = [('net_image', self.net), ('net_mask', self.net_mask),
                    ('net_merge', self.net_merge), ('net_head', self.net_head)]
        for i, (name, net) in enumerate(net_list):
            if net is None:
                continue
            if i != 0:
                net_text += '\n\n\n'
            n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            net_text += '# of ' + name + ' parameters: ' + str(n_params) + '\n\n' + str(net)
        with open(os.path.join(self.output_path, FILENAMES['net_text']), "w") as net_file:
            net_file.write(net_text)

    def save_net_and_optimizer(self, epoch: int = 0):
        self.save_torch_dict(self.optimizer.state_dict(), FILENAMES['optimizer_state'], epoch)
        self.save_torch_dict(self.net.state_dict(), 'net_image.pth', epoch)
        self.save_torch_dict(self.net_merge.state_dict(), 'net_merge.pth', epoch)
        self.save_torch_dict(self.net_head.state_dict(), 'net_head.pth', epoch)
        if self.net_mask is not None:
            self.save_torch_dict(self.net_mask.state_dict(), 'net_mask.pth', epoch)

    def load_net_and_optimizer(self, epoch: int = 0):
        self.optimizer.load_state_dict(self.load_torch_dict(FILENAMES['optimizer_state'], epoch))
        self.net.load_state_dict(self.load_torch_dict('net_image.pth', epoch))
        self.net_merge.load_state_dict(self.load_torch_dict('net_merge.pth', epoch))
        self.net_head.load_state_dict(self.load_torch_dict('net_head.pth', epoch))
        if self.net_mask is not None:
            self.net_mask.load_state_dict(self.load_torch_dict('net_mask.pth', epoch))

    def set_used_config(self) -> list[str]:
        return super().set_used_config() + ['guidednets']

    def pre_meta_train_test(self, epoch: int):
        self.net_merge.train()
        self.net_head.train()
        if self.net_mask is not None:
            self.net_mask.train()

    def meta_train_test_step(self, train_data: TensorDataItem, test_data: TensorDataItem) -> float:
        x_tr, _, y_tr, _ = train_data
        x_ts, y_ts, _, _ = test_data
        if self.config['learn']['use_gpu']:
            x_tr = x_tr.cuda()
            y_tr = y_tr.cuda()
            x_ts = x_ts.cuda()
            y_ts = y_ts.cuda()

        self.net.zero_grad()
        self.net_merge.zero_grad()
        self.net_head.zero_grad()
        if self.net_mask is not None:
            self.net_mask.zero_grad()

        embeddings = self.get_embeddings(x_tr, y_tr)
        prototypes = self.get_prototypes(embeddings, list(x_ts.shape))
        p_ts = self.get_predictions(x_ts, prototypes)

        if self.calc_loss.loss_type == 'mce':
            self.calc_loss.set_mce_weights_from_target(y_tr, self.config['data']['num_classes'])
        loss = self.calc_loss(p_ts, y_ts)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def tune_train_test_process(self, epoch: int,
                                tune_loader: DatasetLoaderItem) -> tuple[list[NDArray], list[NDArray], list[str]]:
        labels, preds, names = [], [], []

        with torch.no_grad():

            self.net.eval()
            self.net_merge.eval()
            self.net_head.eval()
            if self.net_mask is not None:
                self.net_mask.eval()

            self.net.zero_grad()
            self.net_merge.zero_grad()
            self.net_head.zero_grad()
            if self.net_mask is not None:
                self.net_mask.zero_grad()

            embed_tr_list = []
            for i, data in enumerate(tune_loader['train']):
                x_tr, _, y_tr, _ = data
                if self.config['learn']["use_gpu"]:
                    x_tr = x_tr.cuda()
                    y_tr = y_tr.cuda()

                embed_tr = self.get_embeddings(x_tr, y_tr)
                embed_tr_list.append(embed_tr)

            all_embed_tr = torch.vstack(embed_tr_list)
            prototypes = self.get_prototypes(all_embed_tr, [1, 1, x_tr.shape[2], x_tr.shape[3]])

            for i, data in enumerate(tune_loader['test']):
                x_ts, y_ts, _, img_name = data

                if self.config['learn']["use_gpu"]:
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
        image_embedding = self.net(image)
        c_masks = self.one_hot_masks(mask, self.config['data']['num_classes'])
        mask_embeddings = [self.net_mask(c_mask) if self.net_mask is not None else c_mask for c_mask in c_masks]

        if self.net_mask is None:
            # TODO: isn't the combined_mask_embeddings just become ones tensor
            combined_mask_embeddings = torch.zeros_like(mask_embeddings[0])
            for mask_embedding in mask_embeddings:
                combined_mask_embeddings += mask_embedding
            combined_embeddings = image_embedding * combined_mask_embeddings
        else:
            combined_embeddings = torch.clone(image_embedding)
            for mask_embedding in mask_embeddings:
                combined_embeddings *= mask_embedding

        merged_embeddings = self.net_merge(combined_embeddings)

        return merged_embeddings

    @staticmethod
    def get_prototypes(embeddings: Tensor, ref_shape: list[int]) -> Tensor:
        proto = torch.mean(embeddings, dim=0, keepdim=True)
        proto = torch.tile(proto,
                           (ref_shape[0], 1, ref_shape[2], ref_shape[3]))
        return proto

    def get_predictions(self, image: Tensor, prototypes: Tensor) -> Tensor:
        image_embedding = self.net(image)
        mask_embedding = torch.cat([prototypes, image_embedding], dim=1)
        mask = self.net_head(mask_embedding)
        return mask
