from typing import Any
from dataclasses import dataclass


@dataclass
class CIFARData:
    name: str = "cifar"
    path: str = "/data/to/cifar/dataset"
    test_size: float = 0.2
    seed: int = 42


@dataclass
class MNISTData:
    name: str = "mnist"
    path: str = "/data/to/mnist/dataset"
    test_size: float = 0.15
    seed: int = 123


@dataclass
class Model:
    name: str
    layers: int
    output_dim: int


@dataclass
class Training:
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    gpu_id: int
    grad_accumulate_batches: int
    precision: str


@dataclass
class Params:
    data: Any
    model: Model
    training: Training
