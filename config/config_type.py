from typing import TypedDict


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int  # Number of workers on data loader.
    batch_size: int
    resize_to: tuple[int, int]


class DataTuneConfig(TypedDict):
    list_shots: list[int]  # Number of shots (i.e, total annotated samples)
    list_sparsity_point: list[float]  # Number of labeled pixels in point annotation
    list_sparsity_grid: list[float]  # Spacing between pixels in grid annotation
    list_sparsity_contour: list[float]  # Density of the contours (0 to 1)
    list_sparsity_skeleton: list[float]  # Density of the skeletons (0 to 1)
    list_sparsity_region: list[float]  # Ratio of labeled regions (0 to 1)


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
    meta_used_datasets: int  # Number of randomly sampled tasks in meta-learning.
    meta_iterations: int


class SaveConfig(TypedDict):
    ckpt_path: str  # Root folder for checkpoints (model weights)
    output_path: str  # Root folder for general outputs (img predictions, generated train sparse masks, etc)
    exp_name: str
    minimal_save: bool


class WeaselConfig(TypedDict):
    use_first_order: bool  # First order approximation of MAML.
    update_param_rate: float  # MAML inner loop learning rate.
    tune_epochs: int  # Number of epochs on the tuning phase.
    tune_test_freq: int  # Test each tune_test_freq epochs on the tuning phase.


class ProtosegConfig(TypedDict):
    embedding_size: int


class AllConfig(TypedDict):
    data: DataConfig
    data_tune: DataTuneConfig
    learn: LearnConfig
    save: SaveConfig
    weasel: WeaselConfig
    protoseg: ProtosegConfig
