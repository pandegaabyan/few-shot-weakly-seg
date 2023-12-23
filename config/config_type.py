from typing import TypedDict

from data.types import SparsityDict


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int  # Number of workers on data loader.
    batch_size: int
    resize_to: tuple[int, int]


class DataTuneConfig(TypedDict):
    shot_list: list[int]  # Number of shots (i.e, total annotated samples)
    sparsity_dict: SparsityDict  # Sparsity of the annotations
    #   Point: number of labeled pixels in point annotation
    #   Grid: spacing between selected pixels in grid annotation
    #   Contour: density of the contours (1, is the complete contours)
    #   Skeleton: density of the skeletons (1, is the complete skeletons)
    #   Region: percentage of regions labeled (1, all \pure\ regions are labeled)


class LearnConfig(TypedDict):
    should_resume: bool
    use_gpu: bool
    num_epochs: int  # Number of epochs.
    optimizer_lr: float  # Learning rate.
    optimizer_weight_decay: float  # L2 penalty.
    optimizer_momentum: float  # Momentum.
    scheduler_step_size: int
    scheduler_gamma: float
    tune_freq: int  # Run tuning each tune_freq epochs.
    exp_name: str


class WeaselConfig(TypedDict):
    use_first_order: bool  # First order approximation of MAML.
    update_param_step_size: float  # MAML inner loop step size.
    tune_epochs: int  # Number of epochs on the tuning phase.
    tune_test_freq: int  # Test each tune_test_freq epochs on the tuning phase.


class AllConfig(TypedDict):
    data: DataConfig
    data_tune: DataTuneConfig
    learn: LearnConfig
    weasel: WeaselConfig
