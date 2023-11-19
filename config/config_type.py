from typing import TypedDict


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int        # Number of workers on data loader.
    batch_size: int         # Mini-batch size.
    resize_to: tuple[int, int]


class FWSConfig(TypedDict):
    list_shots: list[int]               # Number of shots in the task (i.e, total annotated sparse samples)
    list_sparsity_point: list[float]      # Number of labeled pixels in point annotation
    list_sparsity_grid: list[float]       # Spacing between selected pixels in grid annotation
    list_sparsity_contour: list[float]    # Density of the contours (1, is the complete contours)
    list_sparsity_skeleton: list[float]   # Density of the skeletons (1, is the complete skeletons)
    list_sparsity_region: list[float]     # Percentage of regions labeled (1, all \pure\ regions are labeled)


class TrainConfig(TypedDict):
    use_gpu: bool
    epoch_num: int                  # Number of epochs.
    lr: float                       # Learning rate.
    lr_scheduler_step_size: int
    lr_scheduler_gamma: float
    weight_decay: float             # L2 penalty.
    momentum: float                 # Momentum.
    snapshot: str                   # Starting epoch to resume training. Previously saved weights are loaded.
    test_freq: int                  # Run tuning each test_freq epochs.
    n_metatasks_iter: int           # Number of randomly sampled tasks in meta-learning.


class SaveConfig(TypedDict):
    ckpt_path: str      # Root folder for checkpoints (model weights)
    output_path: str    # Root folder for general outputs (img predictions, generated train sparse masks, etc)
    exp_name: str


class WeaselConfig(TypedDict):
    first_order: bool               # First order approximation of MAML.
    step_size: float                # MAML inner loop step size.
    tuning_epochs: int              # Number of epochs on the tuning phase.
    tuning_freq: int                # Test each tuning_freq epochs on the tuning phase.


class AllConfig(TypedDict):
    data: DataConfig
    fws: FWSConfig
    train: TrainConfig
    save: SaveConfig
    weasel: WeaselConfig
