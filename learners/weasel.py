import time
from collections import OrderedDict

import torch
from numpy.typing import NDArray
from torch.nn import functional

from data.dataset_loaders import DatasetLoaderItem
from data.types import TensorDataItem
from learners.learner import MetaLearner


class WeaselLearner(MetaLearner):

    def set_used_config(self) -> list[str]:
        return super().set_used_config() + ['weasel']

    def meta_train_test_step(self, train_data: TensorDataItem, test_data: TensorDataItem) -> float:
        # Clearing model gradients.
        self.net.zero_grad()

        x_tr, _, y_tr, _ = train_data
        x_ts, y_ts, _, _ = test_data
        if self.config['learn']['use_gpu']:
            x_tr = x_tr.cuda()
            y_tr = y_tr.cuda()
            x_ts = x_ts.cuda()
            y_ts = y_ts.cuda()

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
        outer_loss = functional.cross_entropy(p_ts, y_ts, ignore_index=-1)
        if self.config['learn']["use_gpu"]:
            outer_loss = outer_loss.cuda()

        # Clears the gradients of meta_optimizer.
        self.meta_optimizer.zero_grad()

        # Computing backpropagation.
        outer_loss.backward()
        self.meta_optimizer.step()

        # Returning loss.
        return outer_loss.detach().item()

    def tune_train_test_process(self, epoch: int,
                                tune_loader: DatasetLoaderItem) -> tuple[list[NDArray], list[NDArray], list[str]]:

        # Initiating lists for labels, predictions, and image names.
        labels, preds, names = [], [], []

        # Setting network for training mode.
        self.net.train()

        # Zeroing model gradient.
        self.net.zero_grad()

        # Repeatedly cycling over batches.
        tune_epochs = self.config['weasel']['tune_epochs']
        for c in range(1, tune_epochs + 1):

            self.print_and_log('\tTuning epoch %d/%d' % (c, tune_epochs))

            # Iterating over tune train batches.
            for i, data in enumerate(tune_loader['train']):

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

                labels, preds, names = [], [], []

                with torch.no_grad():

                    start_time = time.time()

                    # Setting network for evaluation mode.
                    self.net.eval()

                    # Iterating over tune test batches.
                    for i, data in enumerate(tune_loader['test']):
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
                        labels.append(y_ts.detach().squeeze(0).cpu().numpy())
                        preds.append(prds)
                        names.append(img_name[0])

                    print_message = f'{c}/{tune_epochs}'
                    score = self.calc_and_log_metrics(labels, preds, print_message, start='\t')

                    end_time = time.time()

                    self.write_score_to_csv(tune_loader, epoch, c,
                                            end_time - start_time, score)

                # Finishing test.

        # Loading model.
        self.load_net_and_optimizer()

        return labels, preds, names

    def update_parameters(self, loss: torch.Tensor):
        grads = torch.autograd.grad(loss, self.net.meta_parameters(),  # type: ignore
                                    create_graph=not self.config['weasel']['use_first_order'])

        params = OrderedDict()
        for (name, param), grad in zip(self.net.meta_named_parameters(), grads):
            params[name] = param - self.config['weasel']['update_param_step_size'] * grad

        return params

    def write_score_to_csv(self, tune_loader: DatasetLoaderItem,
                           epoch: int, tune_epoch: int, test_duration: float, score: dict):
        row = {
            'epoch': epoch,
            'n_shots': tune_loader['n_shots'],
            'sparsity_mode': tune_loader['sparsity_mode'],
            'sparsity_value': tune_loader['sparsity_value'],
            'tune_epoch': tune_epoch,
            'test_duration': test_duration * 10 ** 3
        }
        row.update(score)
        self.write_to_csv(
            'tuning_score.csv',
            ['epoch', 'n_shots', 'sparsity_mode', 'sparsity_value', 'tune_epoch', 'test_duration']
            + sorted(score.keys()),
            row
        )
