import os
from collections import OrderedDict

import numpy as np
import sklearn
import torch
from skimage import io
from torch.nn import functional
from torch.utils.data import DataLoader

from learners.learner import MetaLearner


class WeaselLearner(MetaLearner):
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

            # Acquiring training and test data.
            x_train = []
            y_train = []

            x_test = []
            y_test = []

            for index in indices:
                x_tr, y_tr, x_ts, y_ts = self.prepare_meta_batch(index)

                x_train.append(x_tr)
                y_train.append(y_tr)

                x_test.append(x_ts)
                y_test.append(y_ts)

            ##########################################################################
            # Outer loop. ############################################################
            ##########################################################################

            # Clearing model gradients.
            self.net.zero_grad()

            # Resetting outer loss.
            outer_loss = torch.tensor(0.0)
            if self.config['train']["use_gpu"]:
                outer_loss = outer_loss.cuda()

            # Iterating over tasks.
            for j in range(len(x_train)):
                x_tr = x_train[j]
                y_tr = y_train[j]

                x_ts = x_test[j]
                y_ts = y_test[j]

                ######################################################################
                # Inner loop. ########################################################
                ######################################################################

                # Forwarding through model.
                p_tr = self.net(x_tr)

                # Computing inner loss.
                inner_loss = functional.cross_entropy(p_tr, y_tr, ignore_index=-1)

                # Zeroing model gradient.
                self.net.zero_grad()

                # Computing metaparameters.
                params = self.update_parameters(inner_loss)

                # Verifying performance on task test set.
                p_ts = self.net(x_ts, params=params)

                # Accumulating outer loss.
                outer_loss += functional.cross_entropy(p_ts, y_ts, ignore_index=-1)

                ######################################################################
                # End of inner loop. #################################################
                ######################################################################

            # Clears the gradients of meta_optimizer.
            self.meta_optimizer.zero_grad()

            # Computing loss.
            outer_loss.div_(len(x_test))

            # Computing backpropagation.
            outer_loss.backward()
            self.meta_optimizer.step()

            # Updating loss meter.
            train_outer_loss_list.append(outer_loss.detach().item())

            ##########################################################################
            # End of outer loop. #####################################################
            ##########################################################################

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

        # Setting network for training mode.
        self.net.train()

        # Zeroing model gradient.
        self.net.zero_grad()

        # Repeatedly cycling over batches.
        for c in range(self.config['weasel']['tuning_epochs']):

            print('Tuning epoch %d/%d' % (c + 1, self.config['weasel']['tuning_epochs']))

            # Iterating over tuning train batches.
            for i, data in enumerate(tune_train_loader):

                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_tr, _, y_tr, _ = data

                # Casting to cuda variables.
                if self.config['train']["use_gpu"]:
                    x_tr = x_tr.cuda()
                    y_tr = y_tr.cuda()

                # Zeroing gradients for optimizer.
                self.meta_optimizer.zero_grad()

                # Forwarding through model.
                p_tr = self.net(x_tr)

                # Computing inner loss.
                tune_train_loss = functional.cross_entropy(p_tr, y_tr, ignore_index=-1)

                # Computing gradients and taking step in optimizer.
                tune_train_loss.backward()
                self.meta_optimizer.step()

            if (c + 1) % self.config['weasel']['tuning_freq'] == 0:

                ##########################################
                # Starting test. #########################
                ##########################################

                with torch.no_grad():

                    # Setting network for evaluation mode.
                    self.net.eval()

                    # Initiating lists for labels and predictions.
                    labs_all, prds_all = [], []

                    # Iterating over tuning test batches.
                    for i, data in enumerate(tune_test_loader):
                        # Obtaining images, labels and paths for batch.
                        x_ts, y_ts, _, _ = data

                        # Casting to cuda variables.
                        if self.config['train']["use_gpu"]:
                            x_ts = x_ts.cuda()
                            y_ts = y_ts.cuda()

                        # Forwarding.
                        p_ts = self.net(x_ts)

                        # Obtaining predictions.
                        prds = p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()

                        # Appending data to lists.
                        labs_all.append(y_ts.detach().squeeze(0).cpu().numpy())
                        prds_all.append(prds)

                    # Converting to numpy for computing metrics.
                    labs_np = np.asarray(labs_all).ravel()
                    prds_np = np.asarray(prds_all).ravel()

                    # Computing metrics.
                    iou = sklearn.metrics.jaccard_score(labs_np, prds_np, average="weighted")

                    print('Jaccard test "%s" %d/%d: %.2f' % (
                        sparsity_mode, c + 1, self.config['weasel']['tuning_epochs'], iou * 100,))

                ##########################################
                # Finishing test. ########################
                ##########################################

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
                                           img_name[j] + '_dense.png'), stored_dense)
                    io.imsave(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
                                           sparsity_mode + '_train_epoch_' + str(epoch),
                                           img_name[j] + '_sparse.png'), stored_sparse)

        # List for batch losses.
        tune_test_loss_list = list()

        # Initiating lists for images, labels and predictions.
        labs_all, prds_all = [], []

        with torch.no_grad():

            # Setting network for evaluation mode.
            self.net.eval()

            # Iterating over tuning test batches.
            for i, data in enumerate(tune_test_loader):
                # Obtaining images, labels and paths for batch.
                x_ts, y_ts, _, img_name = data

                # Casting to cuda variables.
                if self.config['train']["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                # Forwarding.
                p_ts = self.net(x_ts)

                # Computing loss.
                tune_test_loss = functional.cross_entropy(p_ts, y_ts, ignore_index=-1)

                # Obtaining predictions.
                prds = p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()

                # Appending data to lists.
                labs_all.append(y_ts.detach().squeeze(0).cpu().numpy())
                prds_all.append(prds)

                # Updating loss meter.
                tune_test_loss_list.append(tune_test_loss.detach().item())

                # Saving predictions.
                if epoch == self.config['train']['epoch_num']:
                    stored_pred = (prds * (255 / prds.max())).astype(np.uint8)
                    io.imsave(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'],
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
        print('Jaccard test "%s" %d/%d: %.2f' % (
            sparsity_mode, self.config['weasel']['tuning_epochs'], self.config['weasel']['tuning_epochs'], iou * 100))
        print('--------------------------------------------------------------------')

        # Loading model.
        self.net.load_state_dict(
            torch.load(os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'meta.pth')))
        self.meta_optimizer.load_state_dict(
            torch.load(os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'opt_meta.pth')))

    def update_parameters(self, loss: torch.Tensor):
        grads = torch.autograd.grad(loss, self.net.meta_parameters(),
                                    create_graph=not self.config['weasel']['first_order'])

        params = OrderedDict()
        for (name, param), grad in zip(self.net.meta_named_parameters(), grads):
            params[name] = param - self.config['weasel']['step_size'] * grad

        return params
