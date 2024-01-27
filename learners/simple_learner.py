import os
import time
from typing import Type

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import io
from torch import nn

from config.config_type import AllConfig
from config.constants import FILENAMES
from data.simple_dataset import SimpleDataset
from data.simple_dataset_loaders import get_simple_dataset_loader
from data.types import SimpleDatasetKeywordArgs
from learners.base_learner import BaseLearner
from learners.losses import CustomLoss
from learners.types import CalcMetrics, Optimizer, Scheduler
from learners.utils import (
    check_mkdir,
    dump_json,
    add_suffix_to_filename,
    get_gpu_memory,
)


class SimpleLearner(BaseLearner):
    def __init__(
        self,
        net: nn.Module,
        config: AllConfig,
        dataset_class: Type[SimpleDataset],
        dataset_kwargs: SimpleDatasetKeywordArgs,
        test_dataset_class: Type[SimpleDataset] | None = None,
        test_dataset_kwargs: SimpleDatasetKeywordArgs | None = None,
        calc_metrics: CalcMetrics | None = None,
        calc_loss: CustomLoss | None = None,
        optimizer: Optimizer | None = None,
        scheduler: Scheduler | None = None,
    ):
        super().__init__(net, config, calc_metrics, calc_loss, optimizer, scheduler)

        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        if test_dataset_class is not None and test_dataset_kwargs is not None:
            self.test_dataset_class = test_dataset_class
            self.test_dataset_kwargs = test_dataset_kwargs
        else:
            self.test_dataset_class = dataset_class
            self.test_dataset_kwargs = dataset_kwargs

        self.dataset_loader = get_simple_dataset_loader(
            self.config["data"],
            self.dataset_class,
            self.dataset_kwargs,
            test_dataset_class=self.test_dataset_class,
            test_dataset_kwargs=self.test_dataset_kwargs,
            pin_memory=self.config["learn"]["use_gpu"],
        )

    @staticmethod
    def set_used_config() -> list[str]:
        return ["data", "learn", "loss", "optimizer", "scheduler"]

    def learn_process(self, epoch: int):
        self.train_val_process(epoch)

        self.save_net_and_optimizer()
        self.update_checkpoint({"epoch": epoch})
        self.scheduler.step()

        num_epochs = self.config["learn"]["num_epochs"]
        test_freq = self.config["learn"].get("test_freq")
        if test_freq is not None and epoch % test_freq == 0 or epoch == num_epochs:
            self.save_net_and_optimizer(epoch)
            self.test_process(epoch)

    def train_val_process(self, epoch: int):
        total_train_loss = 0.0
        total_val_loss = 0.0
        val_labels = []
        val_preds = []

        num_epochs = self.config["learn"]["num_epochs"]
        train_loader_len = len(self.dataset_loader["train"])
        val_loader_len = len(self.dataset_loader["val"])

        start_time = time.time()

        print("")
        self.net.train()
        for i, train_data in enumerate(self.dataset_loader["train"]):
            x_tr, y_tr, _ = train_data
            if self.config["learn"]["use_gpu"]:
                x_tr = x_tr.cuda()
                y_tr = y_tr.cuda()

            self.optimizer.zero_grad()

            p_tr = self.net(x_tr)

            loss = self.calc_loss(p_tr, y_tr)
            loss.backward()
            self.optimizer.step()

            detached_loss = loss.detach().item()
            total_train_loss += detached_loss
            self.print_and_log(
                "Train Ep: %d/%d, it: %d/%d, loss: %.4f"
                % (epoch, num_epochs, i + 1, train_loader_len, detached_loss)
            )
        train_avg_loss = total_train_loss / train_loader_len
        self.print_and_log(
            "Train Ep: %d/%d, avg loss: %.4f" % (epoch, num_epochs, train_avg_loss)
        )

        train_time = time.time()

        self.net.eval()
        with torch.no_grad():
            for i, val_data in enumerate(self.dataset_loader["val"]):
                x_val, y_val, _ = val_data
                if self.config["learn"]["use_gpu"]:
                    x_val = x_val.cuda()
                    y_val = y_val.cuda()

                p_val = self.net(x_val)

                loss = self.calc_loss(p_val, y_val)
                detached_loss = loss.detach().item()
                total_val_loss += detached_loss

                val_labels.append(y_val.detach().cpu().numpy())
                val_preds.append(p_val.detach().max(1)[1].squeeze(1).cpu().numpy())

                self.print_and_log(
                    "Val Ep: %d/%d, it: %d/%d, loss: %.4f"
                    % (epoch, num_epochs, i + 1, val_loader_len, detached_loss)
                )
        val_avg_loss = total_val_loss / val_loader_len
        self.print_and_log(
            "Val Ep: %d/%d, avg loss: %.4f" % (epoch, num_epochs, val_avg_loss)
        )

        val_time = time.time()

        gpu_percent, _ = get_gpu_memory()
        score = self.calc_and_log_metrics(val_labels, val_preds)

        row = {
            "epoch": epoch,
            "train_duration": (train_time - start_time) * 10**3,
            "val_duration": (val_time - train_time) * 10**3,
            "train_avg_loss": train_avg_loss,
            "val_avg_loss": val_avg_loss,
            "post_gpu_percent": gpu_percent - self.initial_gpu_percent,
        }
        row.update(score)
        self.write_to_csv(
            FILENAMES["train_val_loss_score"],
            [
                "epoch",
                "train_duration",
                "val_duration",
                "post_gpu_percent",
                "train_avg_loss",
                "val_avg_loss",
            ]
            + sorted(score.keys()),
            row,
        )

    def test_process(self, epoch: int):
        test_labels = []
        test_preds = []
        test_names = []

        num_epochs = self.config["learn"]["num_epochs"]
        test_loader_len = len(self.dataset_loader["test"])

        start_time = time.time()

        print("")
        self.net.eval()
        with torch.no_grad():
            for i, test_data in enumerate(self.dataset_loader["test"]):
                x_ts, y_ts, img_name = test_data
                if self.config["learn"]["use_gpu"]:
                    x_ts = x_ts.cuda()
                    y_ts = y_ts.cuda()

                p_ts = self.net(x_ts)

                test_labels.append(y_ts.detach().squeeze(0).cpu().numpy())
                test_preds.append(
                    p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
                )
                test_names.append(img_name[0])

                self.print_and_log(
                    "Test Ep: %d, it: %d/%d" % (epoch, i + 1, test_loader_len)
                )

        end_time = time.time()

        gpu_percent, _ = get_gpu_memory()
        score = self.calc_and_log_metrics(test_labels, test_preds)

        row = {
            "dataset": str(self.test_dataset_class).split(".")[-1][:-2],
            "epoch": epoch,
            "duration": (end_time - start_time) * 10**3,
            "post_gpu_percent": gpu_percent - self.initial_gpu_percent,
        }
        row.update(score)
        self.write_to_csv(
            FILENAMES["test_score"],
            ["dataset", "epoch", "duration", "post_gpu_percent"] + sorted(score.keys()),
            row,
        )

        if epoch == num_epochs:
            for pred, name in zip(test_preds, test_names):
                self.save_prediction(pred, name)

    def retest(self, epochs: list[int] | None = None):
        gpu_percent, gpu_total = self.initialize_gpu_usage()

        ok = self.check_output_and_ckpt_dir()
        if not ok:
            print("No data from previous learning")
            return

        if epochs is None:
            num_epochs = self.config["learn"]["num_epochs"]
            test_freq = self.config["learn"].get("test_freq")
            if test_freq is None:
                print("No epochs argument and no test_freq in config")
                return
            else:
                epochs = list(range(test_freq, num_epochs + 1, test_freq))

        self.initialize_log()
        self.save_configuration(False)

        self.log_initial_info()
        self.print_and_log(f"Start retesting on epochs: {epochs} ...")
        if self.config["learn"]["use_gpu"]:
            self.print_and_log(
                "Using GPU with total memory %dMiB, %.2f%% is already used"
                % (gpu_total, gpu_percent)
            )

        for epoch in epochs:
            try:
                self.load_net_and_optimizer(epoch)
            except FileNotFoundError:
                self.print_and_log("Checkpoint not found, continue ...")
                continue
            self.test_process(epoch)

        self.print_and_log("Finish retesting ...", end="\n")
        self.remove_log_handlers()

    def get_dataset_loader_dict(self) -> dict:
        return {
            "dataset_class": str(self.dataset_class).replace("'", ""),
            "dataset_kwargs": self.dataset_kwargs,
            "test_dataset_class": str(self.test_dataset_class).replace("'", ""),
            "test_dataset_kwargs": self.test_dataset_kwargs,
        }

    def save_configuration(self, is_new: bool):
        dataset_params = self.get_dataset_loader_dict()
        optimization_data = self.get_optimization_data_dict()
        config_filepath = os.path.join(self.output_path, FILENAMES["config"])
        dataset_path = os.path.join(self.output_path, FILENAMES["dataset_config"])
        opt_path = os.path.join(self.output_path, FILENAMES["optimization_data"])
        if not is_new:
            i = 1
            while (
                os.path.isfile(config_filepath)
                or os.path.isfile(dataset_path)
                or os.path.isfile(opt_path)
            ):
                i += 1
                config_filepath = add_suffix_to_filename(config_filepath, str(i))
                dataset_path = add_suffix_to_filename(dataset_path, str(i))
                opt_path = add_suffix_to_filename(opt_path, str(i))

        dump_json(config_filepath, self.config)
        dump_json(dataset_path, dataset_params)
        dump_json(opt_path, optimization_data)
        if is_new:
            self.save_net_as_text()

    def save_prediction(self, prediction: NDArray, filename: str):
        check_mkdir(os.path.join(self.output_path, FILENAMES["prediction_folder"]))

        stored_prediction = (prediction * (255 / prediction.max())).astype(np.uint8)
        io.imsave(
            os.path.join(
                self.output_path, FILENAMES["prediction_folder"], f"{filename}.png"
            ),
            stored_prediction,
        )
