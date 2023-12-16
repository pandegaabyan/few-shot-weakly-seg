import torch
from numpy.typing import NDArray
from torch.nn import functional
from torch.utils.data import DataLoader

from learners.learner import MetaLearner


class ProtoSegLearner(MetaLearner):

    def meta_train_test_step(self, dataset_indices: list[int]) -> list[float]:
        loss_list = list()

        for index in dataset_indices:
            # Acquiring training and test data.
            x_tr, y_tr, x_ts, y_ts = self.prepare_meta_batch(index)

            # Concatenating tensors.
            x_train = torch.cat([x_tr], dim=0)
            y_train = torch.cat([y_tr], dim=0)

            x_test = torch.cat([x_ts], dim=0)
            y_test = torch.cat([y_ts], dim=0)

            # Clearing model gradients.
            self.net.zero_grad()

            # Start of prototyping

            emb_train = self.net(x_train)
            emb_test = self.net(x_test)

            emb_train_linear = emb_train.permute(0, 2, 3, 1).view(
                emb_train.size(0), emb_train.size(2) * emb_train.size(3), emb_train.size(1))
            emb_test_linear = emb_test.permute(0, 2, 3, 1).view(
                emb_test.size(0), emb_test.size(2) * emb_test.size(3), emb_test.size(1))

            y_train_linear = y_train.view(y_train.size(0), -1)
            y_test_linear = y_test.view(y_test.size(0), -1)

            prototypes = self.get_prototypes(emb_train_linear,
                                             y_train_linear,
                                             self.config['data']['num_classes'])

            outer_loss = self.prototypical_loss(prototypes,
                                                emb_test_linear,
                                                y_test_linear,
                                                ignore_index=-1)

            # End of prototyping

            # Clears the gradients of meta_optimizer.
            self.meta_optimizer.zero_grad()

            # Computing backpropagation.
            outer_loss.backward()
            self.meta_optimizer.step()

            loss_list.append(outer_loss.detach().item())

        # Returning loss.
        return loss_list

    def tune_train_test_process(self, tune_train_loader: DataLoader, tune_test_loader: DataLoader,
                                epoch: int, sparsity_mode: str) -> tuple[list[NDArray], list[NDArray], list[str]]:

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
            for i, data in enumerate(tune_train_loader):

                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_tr, _, y_tr, _ = data

                # Casting tensors to cuda.
                if self.config['learn']["use_gpu"]:
                    x_tr = x_tr.cuda()
                    y_tr = y_tr.cuda()

                emb_tr = self.net(x_tr)

                emb_train_linear = emb_tr.permute(0, 2, 3, 1).view(
                    emb_tr.size(0), emb_tr.size(2) * emb_tr.size(3), emb_tr.size(1))

                y_train_linear = y_tr.view(y_tr.size(0), -1)

                emb_train_list.append(emb_train_linear)
                y_train_list.append(y_train_linear)

            emb_tr = torch.vstack(emb_train_list)
            y_tr = torch.vstack(y_train_list)

            prototypes = self.get_prototypes(emb_tr, y_tr, self.config['data']['num_classes'])

            # Iterating over tune test batches.
            for i, data in enumerate(tune_test_loader):
                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_ts, y_ts, _, img_name = data

                # Casting tensors to cuda.
                if self.config['learn']["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                emb_ts = self.net(x_ts)

                emb_test_linear = emb_ts.permute(0, 2, 3, 1).view(
                    emb_ts.size(0), emb_ts.size(2) * emb_ts.size(3), emb_ts.size(1))

                p_test_linear = self.get_predictions(prototypes, emb_test_linear)

                p_test = p_test_linear.view(p_test_linear.size(0), y_ts.size(1), y_ts.size(2))

                # Taking mode of predictions.
                p_full, _ = torch.mode(p_test, dim=0)

                labels.append(y_ts.cpu().numpy().squeeze())
                preds.append(p_full.cpu().numpy().squeeze())
                names.append(img_name)

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

    @staticmethod
    def prototypical_loss(prototypes, embeddings, targets, **kwargs):
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
        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                       - embeddings.unsqueeze(1)) ** 2, dim=-1)
        return functional.cross_entropy(-squared_distances, targets, **kwargs)

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
        sq_distances = torch.sum((prototypes.unsqueeze(1)
                                  - embeddings.unsqueeze(2)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)
        return predictions
