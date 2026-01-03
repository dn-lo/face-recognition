"""Retinaface decoder."""

from itertools import product
from math import ceil

import torch

from models.retinaface.config import ModelConfig


class _PriorBox:
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


class Decoder:
    """Decoder for RetinaFace outputs."""

    def __init__(
        self,
        cfg: ModelConfig,
        image_size: tuple[int, int],
        device: torch.device | None = None,
        downscale: float | None = None,
    ) -> None:
        """Initialize Decoder with configuration and image size.

        Args:
            cfg: Model configuration.
            image_size : Size of the input image (height, width).
            device: Device to map the priors to. Defaults to None.
            downscale: Downscale factor applied to the image. Defaults to None.
        """
        self._height, self._width = image_size
        prior_box = _PriorBox(cfg, image_size=image_size)
        prior_outputs = prior_box.forward()
        # Prior boxes, size: [num_priors,4]
        self._priors = prior_outputs.data
        if device:
            self._priors = self._priors.to(device)
        self._downscale = downscale
        # Variances of prior boxes, as a list of two floats.
        self._variances = cfg.variance

    # Adapted from https://github.com/Hakuyume/chainer-ssd
    def decode_boxes(self, loc: torch.Tensor) -> torch.Tensor:
        """Decode locations from predictions using priors.

        Undo the encoding we did for offset regression at train time.

        Args:
            loc: locations predictions for loc layers. Shape: [num_priors, 10]
            variances: (list[float]) Variances of prior boxes.

        Return:
            decoded bounding box predictions
        """
        # Prior boxes in center-offset form. Shape: [num_priors,4]
        boxes = torch.cat(
            (
                self._priors[:, :2] + loc[:, :2] * self._variances[0] * self._priors[:, 2:],
                self._priors[:, 2:] * torch.exp(loc[:, 2:] * self._variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        scale = torch.tensor([self._width, self._height], device=boxes.device).repeat(2)
        boxes = boxes * scale
        return boxes * self._downscale if self._downscale else boxes

    def decode_landmarks(self, pre: torch.Tensor) -> torch.Tensor:
        """Decode landmarks from predictions using priors.

        Undoes the encoding we did for offset regression at train time.

        Args:
            pre: landmark predictions for loc layers. Shape: [num_priors,10]

        Return:
            decoded landmark predictions
        """
        landmarks = torch.cat(
            (
                self._priors[:, :2] + pre[:, :2] * self._variances[0] * self._priors[:, 2:],
                self._priors[:, :2] + pre[:, 2:4] * self._variances[0] * self._priors[:, 2:],
                self._priors[:, :2] + pre[:, 4:6] * self._variances[0] * self._priors[:, 2:],
                self._priors[:, :2] + pre[:, 6:8] * self._variances[0] * self._priors[:, 2:],
                self._priors[:, :2] + pre[:, 8:10] * self._variances[0] * self._priors[:, 2:],
            ),
            dim=1,
        )
        scale = torch.tensor([self._width, self._height], device=landmarks.device).repeat(5)
        landmarks = landmarks * scale
        return landmarks * self._downscale if self._downscale else landmarks
