import torch
from numpy.typing import NDArray
from torch import nn

from config.config_type import AllConfig
from data.dataset_loaders import DatasetLoaderItem, DatasetLoaderParamReduced
from data.types import TensorDataItem
from learners.losses import CustomLoss
from learners.meta_learner import MetaLearner
from learners.types import CalcMetrics, NeuralNetworks, Optimizer, Scheduler


class ProtoSegLearner(MetaLearner):
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
        assert isinstance(net, nn.Module), "net should be nn.Module"
        self.net = net

    def set_used_config(self) -> list[str]:
        return super().set_used_config() + ["protoseg"]

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

        # Clearing model gradients.
        self.net.zero_grad()

        # Start of prototyping

        emb_tr = self.net(x_tr)
        emb_ts = self.net(x_ts)

        emb_tr_linear = emb_tr.permute(0, 2, 3, 1).view(
            emb_tr.size(0), emb_tr.size(2) * emb_tr.size(3), emb_tr.size(1)
        )
        emb_ts_linear = emb_ts.permute(0, 2, 3, 1).view(
            emb_ts.size(0), emb_ts.size(2) * emb_ts.size(3), emb_ts.size(1)
        )

        y_tr_linear = y_tr.view(y_tr.size(0), -1)
        y_ts_linear = y_ts.view(y_ts.size(0), -1)

        prototypes = self.get_prototypes(
            emb_tr_linear, y_tr_linear, self.config["data"]["num_classes"]
        )

        loss = self.prototypical_loss(prototypes, emb_ts_linear, y_ts_linear)

        # End of prototyping

        # Clears the gradients of meta_optimizer.
        self.optimizer.zero_grad()

        # Computing backpropagation.
        loss.backward()
        self.optimizer.step()

        # Returning loss.
        return loss.detach().item()

    def tune_train_test_process(
        self, epoch: int, tune_loader: DatasetLoaderItem
    ) -> tuple[list[NDArray], list[NDArray], list[str]]:
        # Initiating lists for labels, predictions, and image names.
        labels, preds, names = [], [], []

        with torch.no_grad():
            # Setting network for training mode.
            self.net.eval()

            # Zeroing model gradient.
            self.net.zero_grad()

            # Creating lists for tune train embeddings and labels.
            emb_train_list = []
            y_train_list = []

            # Iterating over tune train batches.
            for i, data in enumerate(tune_loader["train"]):
                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_tr, _, y_tr, _ = data

                # Casting tensors to cuda.
                if self.config["learn"]["use_gpu"]:
                    x_tr = x_tr.cuda()
                    y_tr = y_tr.cuda()

                emb_tr = self.net(x_tr)

                emb_train_linear = emb_tr.permute(0, 2, 3, 1).view(
                    emb_tr.size(0), emb_tr.size(2) * emb_tr.size(3), emb_tr.size(1)
                )

                y_train_linear = y_tr.view(y_tr.size(0), -1)

                emb_train_list.append(emb_train_linear)
                y_train_list.append(y_train_linear)

            emb_tr = torch.vstack(emb_train_list)
            y_tr = torch.vstack(y_train_list)

            prototypes = self.get_prototypes(
                emb_tr, y_tr, self.config["data"]["num_classes"]
            )

            # Iterating over tune test batches.
            for i, data in enumerate(tune_loader["test"]):
                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_ts, y_ts, _, img_name = data

                # Casting tensors to cuda.
                if self.config["learn"]["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                emb_ts = self.net(x_ts)

                emb_test_linear = emb_ts.permute(0, 2, 3, 1).view(
                    emb_ts.size(0), emb_ts.size(2) * emb_ts.size(3), emb_ts.size(1)
                )

                p_test_linear = self.get_predictions(prototypes, emb_test_linear)

                p_test = p_test_linear.view(
                    p_test_linear.size(0), y_ts.size(1), y_ts.size(2)
                )

                # Taking mode of predictions.
                p_full, _ = torch.mode(p_test, dim=0)

                labels.append(y_ts.cpu().numpy().squeeze())
                preds.append(p_full.cpu().numpy().squeeze())
                names.append(img_name[0])

        return labels, preds, names

    @staticmethod
    def get_num_samples(targets, num_classes, dtype=None):
        """Get the number of annotated pixels for each class"""
        batch_size = targets.size(0)

        with torch.no_grad():
            num_samples = targets.new_zeros((batch_size, num_classes), dtype=dtype)

            for i in range(batch_size):
                trg_i = targets[i]

                for c in range(num_classes):
                    num_samples[i, c] += trg_i[trg_i == c].size(0)

        return num_samples

    def get_prototypes(self, embeddings, targets, num_classes):
        """Compute the prototypes (the mean vector of the embedded training/support
        points belonging to its class) for each classes in the task.
        Parameters
        ----------
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the support points. This tensor
            has shape `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the support points. This tensor has
            shape `(batch_size, num_examples)`.
        num_classes : int
            Number of classes in the task.
        Returns
        -------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        """
        batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

        num_samples = self.get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
        indices = targets.unsqueeze(-1).expand_as(embeddings)

        for i in range(indices.size(0)):
            trg_i = targets[i]
            emb_i = embeddings[i]

            for c in range(num_classes):
                prototypes[i, c] += torch.sum(emb_i[trg_i == c], dim=0)

        prototypes.div_(num_samples)

        return prototypes

    def prototypical_loss(self, prototypes, embeddings, targets):
        """Compute the loss (i.e. negative log-likelihood) for the prototypical
        network, on the test/query points.
        Parameters
        ----------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the query points. This tensor has
            shape `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the query points. This tensor has
            shape `(batch_size, num_examples)`.
        Returns
        -------
        loss : `torch.FloatTensor` instance
            The negative log-likelihood on the query points.
        """
        batch_diff = prototypes.shape[0] - embeddings.shape[0]
        if batch_diff == 0:
            new_embed = embeddings
            new_target = targets
        elif batch_diff > 0:
            new_embed = torch.cat([embeddings, embeddings[:batch_diff]], dim=0)
            new_target = torch.cat([targets, targets[:batch_diff]], dim=0)
        else:
            new_embed = embeddings[: prototypes.shape[0]]
            new_target = targets[: prototypes.shape[0]]
        squared_distances = torch.sum(
            (prototypes.unsqueeze(2) - new_embed.unsqueeze(1)) ** 2, dim=-1
        )
        return self.calc_loss(-squared_distances, new_target)

    @staticmethod
    def get_predictions(prototypes, embeddings):
        """Compute the accuracy of the prototypical network on the test/query points.
        Parameters
        ----------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(meta_batch_size, num_classes, embedding_size)`.
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the query points. This tensor has
            shape `(meta_batch_size, num_examples, embedding_size)`.
        Returns
        -------
        accuracy : `torch.FloatTensor` instance
            Mean accuracy on the query points.
        """
        squared_distances = torch.sum(
            (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
        )
        predictions = torch.argmin(squared_distances, dim=-1)
        return predictions
