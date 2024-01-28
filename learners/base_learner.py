import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn import metrics
from torch import nn, optim

from config.config_type import ConfigBase, ConfigUnion
from config.constants import DEFAULT_CONFIGS, FILENAMES
from learners.losses import CustomLoss
from learners.types import CalcMetrics, NeuralNetworks, Optimizer, Scheduler
from learners.utils import (
    check_mkdir,
    check_rmtree,
    dump_json,
    get_gpu_memory,
    get_name_from_function,
    get_name_from_instance,
    get_short_git_hash,
    get_simple_stack_list,
    load_json,
)


class BaseLearner(ABC):
    def __init__(
        self,
        net: NeuralNetworks,
        config: ConfigBase,
        calc_metrics: CalcMetrics | None = None,
        calc_loss: CustomLoss | None = None,
        optimizer: Optimizer | None = None,
        scheduler: Scheduler | None = None,
    ):
        self.net = net
        self.calc_metrics = calc_metrics
        self.config = config

        self.output_path = os.path.join(
            FILENAMES["output_folder"], self.config["learn"]["exp_name"]
        )
        self.ckpt_path = os.path.join(
            FILENAMES["checkpoint_folder"], self.config["learn"]["exp_name"]
        )

        self.checkpoint = {}
        self.initial_gpu_percent = 0
        self.initial_messages = []

        if calc_loss is not None:
            self.calc_loss = calc_loss
        else:
            self.calc_loss = CustomLoss()

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if isinstance(net, nn.Module):
                net_params = net.parameters()
            else:
                net_params = []
                for n in net.values():
                    net_params += n.parameters()
            self.optimizer = optim.Adam(
                [
                    {
                        "params": net_params,
                        "lr": self.config["optimizer"].get(
                            "lr", DEFAULT_CONFIGS["optimizer_lr"]
                        ),
                    }
                ]
            )

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                self.config["scheduler"].get(
                    "step_size", DEFAULT_CONFIGS["scheduler_step_size"]
                ),
            )

    @abstractmethod
    def save_configuration(self, is_new: bool):
        pass

    @abstractmethod
    def learn_process(self, epoch: int):
        pass

    def learn(self):
        gpu_percent, gpu_total = self.initialize_gpu_usage()

        # Loading optimizer state in case of resuming training.
        if self.config["learn"]["should_resume"]:
            ok = self.check_output_and_ckpt_dir()
            if not ok:
                print("No data from previous learning")
                return

            self.read_checkpoint()
            self.load_net_and_optimizer()

            self.initialize_log()
            self.save_configuration(False)

            self.log_initial_info()
            self.print_and_log("Resume learning ...")
            curr_epoch = self.checkpoint["epoch"] + 1

        else:
            ok = self.clear_output_and_ckpt_dir()
            if not ok:
                print("Learning canceled")
                return

            self.create_output_and_ckpt_dir()

            self.initialize_log()
            self.save_configuration(True)

            self.log_initial_info()
            self.print_and_log("Start learning ...")
            curr_epoch = 1

        if self.config["learn"]["use_gpu"]:
            self.print_and_log(
                "Using GPU with total memory %dMiB, %.2f%% is already used"
                % (gpu_total, gpu_percent)
            )

        # Iterating over epochs.
        for epoch in range(curr_epoch, self.config["learn"]["num_epochs"] + 1):
            self.learn_process(epoch)

        self.print_and_log("Finish learning ...", end="\n")
        self.remove_log_handlers()

    def check_and_clean_config(self, config: ConfigUnion, ref_type: Type[ConfigUnion]):
        ref_keys = ref_type.__annotations__.keys()
        for key in config.keys():
            if key not in ref_keys:
                del config[key]
        for key in ref_keys:
            if key not in config.keys():
                raise ValueError(f"Missing {key} in config")

    def initialize_gpu_usage(self) -> tuple[float, int]:
        gpu_percent, gpu_total = 0, 0
        if self.config["learn"]["use_gpu"]:
            gpu_percent, gpu_total = get_gpu_memory()
            self.initial_gpu_percent = gpu_percent
            if isinstance(self.net, nn.Module):
                self.net = self.net.cuda()
            else:
                for net in self.net.values():
                    net.cuda()
        return gpu_percent, gpu_total

    def check_output_and_ckpt_dir(self) -> bool:
        return os.path.exists(self.output_path) and os.path.exists(self.ckpt_path)

    def create_output_and_ckpt_dir(self):
        check_mkdir(FILENAMES["output_folder"])
        check_mkdir(self.output_path)
        check_mkdir(FILENAMES["checkpoint_folder"])
        check_mkdir(self.ckpt_path)

    def clear_output_and_ckpt_dir(self) -> bool:
        output_ok = check_rmtree(self.output_path)
        ckpt_ok = check_rmtree(self.ckpt_path)
        return output_ok and ckpt_ok

    def initialize_log(self):
        logging.basicConfig(
            filename=os.path.join(self.output_path, FILENAMES["learn_log"]),
            encoding="utf-8",
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    @staticmethod
    def print_and_log(message: str, start: str = "", end: str = ""):
        print(start + message + end)
        logging.info(start.replace("\n", "") + message + end.replace("\n", ""))

    @staticmethod
    def log_error():
        logging.error("Exception:", exc_info=True, stack_info=True)

    @staticmethod
    def remove_log_handlers():
        logger = logging.getLogger()
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    def write_to_csv(self, filename: str, fieldnames: list[str], row: dict):
        filename = os.path.join(self.output_path, filename)
        if os.path.isfile(filename):
            with open(filename, "a", encoding="UTF8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        else:
            with open(filename, "w", encoding="UTF8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

    def read_checkpoint(self):
        self.checkpoint = load_json(
            os.path.join(self.ckpt_path, FILENAMES["checkpoint"])
        )

    def update_checkpoint(self, data: dict):
        self.checkpoint.update(data)
        dump_json(
            os.path.join(self.ckpt_path, FILENAMES["checkpoint"]), self.checkpoint
        )

    def log_initial_info(self):
        self.print_and_log("-" * 30)
        self.print_and_log("Git hash: " + get_short_git_hash())
        stack_list = get_simple_stack_list(end=-3)
        if "ipykernel" in stack_list[-1]:
            self.print_and_log("Call stack: (called in jupyter notebook)")
        else:
            for i, stack in enumerate(stack_list):
                self.print_and_log(f"Call stack ({i}): {stack}")
        for msg in self.initial_messages:
            self.print_and_log("Note: " + msg)
        print("")

    def set_initial_messages(self, messages: list[str]):
        self.initial_messages = messages

    def get_optimization_data_dict(self) -> dict:
        optimizer_data = [
            {
                key: param_group[key]
                for key in filter(lambda x: x != "params", param_group.keys())
            }
            for param_group in self.optimizer.__dict__["param_groups"]
        ]
        scheduler_data = {
            key: self.scheduler.__dict__[key]
            for key in filter(
                lambda x: x != "optimizer" and not x.startswith("_"),
                self.scheduler.__dict__.keys(),
            )
        }
        return {
            "metrics": None
            if self.calc_metrics is None
            else get_name_from_function(self.calc_metrics),
            "loss": get_name_from_instance(self.calc_loss),
            "loss_data": self.calc_loss.params(),
            "optimizer": get_name_from_instance(self.optimizer),
            "optimizer_data": optimizer_data,
            "scheduler": get_name_from_instance(self.scheduler),
            "scheduler_data": scheduler_data,
        }

    def save_net_as_text(self):
        if isinstance(self.net, nn.Module):
            n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            net_text = "# of parameters: " + str(n_params) + "\n\n" + str(self.net)
        else:
            net_text = ""
            for i, name in enumerate(self.net.keys()):
                if i != 0:
                    net_text += "\n\n\n"
                net = self.net[name]
                n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                net_text += (
                    "# of net_"
                    + name
                    + " parameters: "
                    + str(n_params)
                    + "\n\n"
                    + str(net)
                )
        with open(
            os.path.join(self.output_path, FILENAMES["net_text"]), "w"
        ) as net_file:
            net_file.write(net_text)

    def calc_and_log_metrics(
        self,
        labels: list[NDArray],
        preds: list[NDArray],
        message: str = "",
        start: str = "",
        end: str = "",
    ) -> dict:
        if self.calc_metrics is None:
            iou_mean = metrics.jaccard_score(
                np.concatenate(labels, axis=0).ravel(),
                np.concatenate(preds, axis=0).ravel(),
                average="macro",
            )
            iou_mean = float(iou_mean)
            score = {"iou_mean": iou_mean}
            score_text = "%.2f" % (iou_mean * 100)
            name = "Mean IoU score"
        else:
            score, score_text, name = self.calc_metrics(labels, preds)
        if message == "":
            full_message = f"{name}: {score_text}"
        else:
            full_message = f"{name} - {message}: {score_text}"
        self.print_and_log(full_message, start, end)
        return score

    def save_torch_dict(self, state_dict: dict, filename: str, epoch: int = 0):
        prefix = f"ep{epoch}_" if epoch != 0 else ""
        torch.save(state_dict, os.path.join(self.ckpt_path, prefix + filename))

    def load_torch_dict(self, filename: str, epoch: int = 0) -> dict:
        prefix = f"ep{epoch}_" if epoch != 0 else ""
        return torch.load(os.path.join(self.ckpt_path, prefix + filename))

    def save_net_and_optimizer(self, epoch: int = 0):
        self.save_torch_dict(
            self.optimizer.state_dict(), FILENAMES["optimizer_state"], epoch
        )
        if isinstance(self.net, nn.Module):
            self.save_torch_dict(self.net.state_dict(), FILENAMES["net_state"], epoch)
        else:
            for name, net in self.net.items():
                self.save_torch_dict(net.state_dict(), f"net_{name}.pth", epoch)

    def load_net_and_optimizer(self, epoch: int = 0):
        self.optimizer.load_state_dict(
            self.load_torch_dict(FILENAMES["optimizer_state"], epoch)
        )
        if isinstance(self.net, nn.Module):
            self.net.load_state_dict(
                self.load_torch_dict(FILENAMES["net_state"], epoch)
            )
        else:
            for name, net in self.net.items():
                net.load_state_dict(self.load_torch_dict(f"net_{name}.pth", epoch))
