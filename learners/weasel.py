from collections import OrderedDict

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from learners.learner import MetaLearner


class WeaselLearner(MetaLearner):

    def meta_train_step(self, dataset_indices: list[int]) -> list[float]:
        # Acquiring training and test data.

        x_train = []
        y_train = []

        x_test = []
        y_test = []

        for index in dataset_indices:
            x_tr, y_tr, x_ts, y_ts = self.prepare_meta_batch(index)

            x_train.append(x_tr)
            y_train.append(y_tr)

            x_test.append(x_ts)
            y_test.append(y_ts)

        # Clearing model gradients.
        self.net.zero_grad()

        # Resetting outer loss.
        outer_loss = torch.tensor(0.0)
        if self.config['learn']["use_gpu"]:
            outer_loss = outer_loss.cuda()

        # Iterating over datasets.
        for j in range(len(x_train)):
            x_tr = x_train[j]
            y_tr = y_train[j]

            x_ts = x_test[j]
            y_ts = y_test[j]

            # Forwarding through model.
            p_tr = self.net(x_tr)

            # Computing inner loss.
            inner_loss = functional.cross_entropy(p_tr, y_tr, ignore_index=-1)

            # Zeroing model gradient.
            self.net.zero_grad()

            # Computing metaparameters.
            params = self.update_parameters(inner_loss)

            # Verifying performance on test set.
            p_ts = self.net(x_ts, params=params)

            # Accumulating outer loss.
            outer_loss += functional.cross_entropy(p_ts, y_ts, ignore_index=-1)

        # Clears the gradients of meta_optimizer.
        self.meta_optimizer.zero_grad()

        # Computing loss.
        outer_loss.div_(len(x_test))

        # Computing backpropagation.
        outer_loss.backward()
        self.meta_optimizer.step()

        # Returning loss.
        return [outer_loss.detach().item()]

    def tune_train_test(self, tune_train_loader: DataLoader, tune_test_loader: DataLoader,
                        epoch: int, sparsity_mode: str):

        # Setting network for training mode.
        self.net.train()

        # Zeroing model gradient.
        self.net.zero_grad()

        # Repeatedly cycling over batches.
        tune_epochs = self.config['weasel']['tune_epochs']
        for c in range(1, tune_epochs + 1):

            print('Tuning epoch %d/%d' % (c, tune_epochs))

            # Iterating over tune train batches.
            for i, data in enumerate(tune_train_loader):

                # Obtaining images, dense labels, sparse labels and paths for batch.
                x_tr, _, y_tr, _ = data

                # Casting to cuda variables.
                if self.config['learn']["use_gpu"]:
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

            if (c % self.config['weasel']['tune_test_freq'] == 0
                    or c == tune_epochs):

                # Starting test.

                with torch.no_grad():

                    # Setting network for evaluation mode.
                    self.net.eval()

                    # Initiating lists for labels and predictions.
                    labs_all, prds_all = [], []

                    # Iterating over tune test batches.
                    for i, data in enumerate(tune_test_loader):
                        # Obtaining images, labels and paths for batch.
                        x_ts, y_ts, _, img_name = data

                        # Casting to cuda variables.
                        if self.config['learn']["use_gpu"]:
                            x_ts = x_ts.cuda()
                            y_ts = y_ts.cuda()

                        # Forwarding.
                        p_ts = self.net(x_ts)

                        # Obtaining predictions.
                        prds = p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()

                        # Appending data to lists.
                        labs_all.append(y_ts.detach().squeeze(0).cpu().numpy())
                        prds_all.append(prds)

                    print_message = f'"{sparsity_mode}" {c}/{tune_epochs}'
                    self.calc_print_metrics(labs_all, prds_all, print_message)

                    # Saving predictions.
                    if epoch == self.config['learn']['num_epochs'] and c == tune_epochs:
                        self.save_prediction(prds, img_name[0],
                                             sparsity_mode)

                # Finishing test.

        # Loading model.
        self.load_net_and_optimizer()

    def update_parameters(self, loss: torch.Tensor):
        grads = torch.autograd.grad(loss, self.net.meta_parameters(),
                                    create_graph=not self.config['weasel']['use_first_order'])

        params = OrderedDict()
        for (name, param), grad in zip(self.net.meta_named_parameters(), grads):
            params[name] = param - self.config['weasel']['update_param_step_size'] * grad

        return params
