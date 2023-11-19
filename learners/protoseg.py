import os

import numpy as np
import sklearn
import torch
from skimage import io
from torch.nn import functional
from torch.utils.data import DataLoader

from learners.learner import MetaLearner


class ProtoSegLearner(MetaLearner):
    def meta_train_test(self, epoch: int):

        # Setting network for training mode.
        self.net.train()

        # List for batch losses.
        train_outer_loss_list = list()

        num_tasks = len(self.meta_set['train'])

        n_batches = 5

        # Iterating over batches.
        for i in range(n_batches):

            # Randomly selecting tasks.
            perm = np.random.permutation(num_tasks)
            print('Ep: ' + str(epoch) + ', it: ' + str(i + 1) + ', task subset: ' + str(
                perm[:self.config['train']['n_metatasks_iter']]))

            indices = perm[:self.config['train']['n_metatasks_iter']]

            for index in indices:
                # Acquiring training and test data.
                x_train = []
                y_train = []

                x_test = []
                y_test = []

                x_tr, y_tr, x_ts, y_ts = self.prepare_meta_batch(index)

                x_train.append(x_tr)
                y_train.append(y_tr)

                x_test.append(x_ts)
                y_test.append(y_ts)

                # Concatenating tensors.
                x_train = torch.cat(x_train, dim=0)
                y_train = torch.cat(y_train, dim=0)

                x_test = torch.cat(x_test, dim=0)
                y_test = torch.cat(y_test, dim=0)

                # Clearing model gradients.
                self.net.zero_grad()

                ##########################################################################
                # Start of prototyping. ##################################################
                ##########################################################################

                emb_train = self.net(x_train)
                emb_test = self.net(x_test)

                emb_train_linear = emb_train.permute(0, 2, 3, 1).view(
                    emb_train.size(0), emb_train.size(2) * emb_train.size(3), emb_train.size(1))
                emb_test_linear = emb_test.permute(0, 2, 3, 1).view(
                    emb_test.size(0), emb_test.size(2) * emb_test.size(3), emb_test.size(1))

                y_train_linear = y_train.view(y_train.size(0), -1)
                y_test_linear = y_test.view(y_test.size(0), -1)

                prototypes = get_prototypes(emb_train_linear,
                                            y_train_linear,
                                            self.config['data']['num_classes'])

                outer_loss = prototypical_loss(prototypes,
                                               emb_test_linear,
                                               y_test_linear,
                                               ignore_index=-1)

                ##########################################################################
                # End of prototyping. ####################################################
                ##########################################################################

                # Clears the gradients of meta_optimizer.
                self.meta_optimizer.zero_grad()

                # Computing backpropagation.
                outer_loss.backward()
                self.meta_optimizer.step()

                # Updating loss meter.
                train_outer_loss_list.append(outer_loss.detach().item())

        # Saving meta-model.
        if epoch % self.config['train']['test_freq'] == 0:
            torch.save(self.net.state_dict(),
                       os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'meta.pth'))
            torch.save(self.meta_optimizer.state_dict(),
                       os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'opt_meta.pth'))

        # Printing epoch loss.
        print('--------------------------------------------------------------------')
        print('[epoch %d], [train loss %.4f]' % (
            epoch, np.asarray(train_outer_loss_list).mean()))
        print('--------------------------------------------------------------------')

    def tune_train_test(self, tune_train_loader: DataLoader, tune_test_loader: DataLoader,
                        epoch: int, sparsity_mode: str):

        # Creating output directories.
        if epoch == self.config['train']['epoch_num']:
            self.check_mkdir(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                          sparsity_mode + '_train_epoch_' + str(epoch)))
            self.check_mkdir(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                          sparsity_mode + '_test_epoch_' + str(epoch)))

        with torch.no_grad():

            # Setting network for training mode.
            self.net.eval()

            # Zeroing model gradient.
            self.net.zero_grad()

            # Creating lists for tune train embeddings and labels.
            emb_train_list = []
            y_train_list = []

            # Iterating over tuning train batches.
            for i, data in enumerate(tune_train_loader):

                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_tr, _, y_tr, _ = data

                # Casting tensors to cuda.
                if self.config['train']["use_gpu"]:
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

            prototypes = get_prototypes(emb_tr, y_tr, self.config['data']['num_classes'])

            # Lists for whole epoch loss.
            labs_all, prds_all = [], []

            # Iterating over tuning test batches.
            for i, data in enumerate(tune_test_loader):
                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_ts, y_ts, _, img_name = data

                # Casting tensors to cuda.
                if self.config['train']["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                emb_ts = self.net(x_ts)

                emb_test_linear = emb_ts.permute(0, 2, 3, 1).view(
                    emb_ts.size(0), emb_ts.size(2) * emb_ts.size(3), emb_ts.size(1))

                p_test_linear = get_predictions(prototypes, emb_test_linear)

                p_test = p_test_linear.view(p_test_linear.size(0), y_ts.size(1), y_ts.size(2))

                # Taking mode of predictions.
                p_full, _ = torch.mode(p_test, dim=0)

                labs_all.append(y_ts.cpu().numpy().squeeze())
                prds_all.append(p_full.cpu().numpy().squeeze())

                # Saving predictions.
                if epoch == self.config['train']['epoch_num']:
                    stored_pred = p_full.cpu().numpy().squeeze()
                    stored_pred = (stored_pred * (255 / stored_pred.max())).astype(np.uint8)
                    io.imsave(
                        os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                     sparsity_mode + '_test_epoch_' + str(epoch),
                                     img_name[0] + '.png'),
                        stored_pred)

        # Converting to numpy for computing metrics.
        labs_np = np.asarray(labs_all).ravel()
        prds_np = np.asarray(prds_all).ravel()

        # Computing metrics.
        iou = sklearn.metrics.jaccard_score(labs_np, prds_np, average="weighted")

        # Printing metric.
        print('--------------------------------------------------------------------')
        print('Jaccard test "%s": %.2f' % (sparsity_mode, iou * 100))
        print('--------------------------------------------------------------------')

        if epoch == self.config['train']['epoch_num']:

            # Iterating over tuning train batches for saving.
            for i, data in enumerate(tune_train_loader):

                # Obtaining images, dense labels, sparse labels and paths for batch.
                _, y_dense, y_sparse, img_name = data

                for j in range(len(img_name)):
                    stored_dense = y_dense[j].cpu().squeeze().numpy()
                    stored_dense = (stored_dense * (255 / stored_dense.max())).astype(np.uint8)
                    stored_sparse = y_sparse[j].cpu().squeeze().numpy() + 1
                    stored_sparse = (stored_sparse * (255 / stored_sparse.max())).astype(np.uint8)
                    io.imsave(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                           sparsity_mode + '_train_epoch_' + str(epoch),
                                           img_name[j] + '_dense.png'),
                              stored_dense)
                    io.imsave(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                           sparsity_mode + '_train_epoch_' + str(epoch),
                                           img_name[j] + '_sparse.png'),
                              stored_sparse)


# Function to get the number of annotated pixels for each class
def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)

    with torch.no_grad():
        num_samples = targets.new_zeros((batch_size, num_classes), dtype=dtype)

        for i in range(batch_size):
            trg_i = targets[i]

            for c in range(num_classes):
                num_samples[i, c] += trg_i[trg_i == c].size(0)

    return num_samples


def get_prototypes(embeddings, targets, num_classes):
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

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
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
