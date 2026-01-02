"""Layers for RetinaFace model."""

from itertools import product
from math import ceil

import torch

from models.retinaface.config import ModelConfig


class PriorBox:
    """Class to compute prior box coordinates in center-offset form for each source feature map."""

    def __init__(self, cfg: ModelConfig, image_size: tuple[int, int] | None = None) -> None:
        """Initialize PriorBox with configuration.

        Args:
            cfg: Model configuration.
            image_size : Size of the input image (height, width). Defaults to None.
            phase: Phase of operation ('train' or 'test'). Defaults to "train".
        """
        self.min_sizes = cfg.min_sizes
        self.steps = cfg.steps
        self.clip = cfg.clip
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]
        self.name = "s"

    def forward(self) -> torch.Tensor:
        """Compute priorbox coordinates.

        Returns:
            torch.Tensor: Tensor with size [N, 4] containing priorbox coordinates, where N is the
                number of boxes.
        """
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
