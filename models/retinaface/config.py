"""Configs for RetinaFace model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Retinaface model."""

    name: str
    min_sizes: list[list[int]]
    steps: list[int]
    variance: list[float]
    clip: bool
    loc_weight: float
    gpu_train: bool
    batch_size: int
    ngpu: int
    epoch: int
    decay1: int
    decay2: int
    image_size: int
    pretrain: bool
    return_layers: dict[str, int]
    in_channel: int
    out_channel: int


# Configuration instances
CONFIG_MOBILE_NET = ModelConfig(
    name="mobilenet0.25",
    min_sizes=[[16, 32], [64, 128], [256, 512]],
    steps=[8, 16, 32],
    variance=[0.1, 0.2],
    clip=False,
    loc_weight=2.0,
    gpu_train=True,
    batch_size=32,
    ngpu=1,
    epoch=250,
    decay1=190,
    decay2=220,
    image_size=640,
    pretrain=True,
    return_layers={"stage1": 1, "stage2": 2, "stage3": 3},
    in_channel=32,
    out_channel=64,
)

CONFIG_RESNET_50 = ModelConfig(
    name="Resnet50",
    min_sizes=[[16, 32], [64, 128], [256, 512]],
    steps=[8, 16, 32],
    variance=[0.1, 0.2],
    clip=False,
    loc_weight=2.0,
    gpu_train=True,
    batch_size=24,
    ngpu=4,
    epoch=100,
    decay1=70,
    decay2=90,
    image_size=840,
    pretrain=True,
    return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
    in_channel=256,
    out_channel=256,
)
