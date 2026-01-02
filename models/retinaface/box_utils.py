"""Utility functions for bounding box operations."""

import torch


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(
    loc: torch.Tensor, priors: torch.Tensor, variances: list[float], hw: tuple[int, int]
) -> torch.Tensor:
    """Decode locations from predictions using priors.

    Undo the encoding we did for offset regression at train time.

    Args:
        loc: locations predictions for loc layers. Shape: [num_priors,10]
        priors: Prior boxes in center-offset form. Shape: [num_priors,4].
        variances: (list[float]) Variances of prior boxes.
        hw: (tuple[int, int]) height and width of the image.

    Return:
        decoded bounding box predictions
    """
    height, width = hw
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    scale = torch.tensor([width, height], device=boxes.device).repeat(2)
    return boxes * scale


def decode_landmarks(
    pre: torch.Tensor, priors: torch.Tensor, variances: list[float], hw: tuple[int, int]
) -> torch.Tensor:
    """Decode landmarks from predictions using priors.

    Undoes the encoding we did for offset regression at train time.

    Args:
        pre: landmark predictions for loc layers. Shape: [num_priors,10]
        priors: Prior boxes in center-offset form. Shape: [num_priors,4].
        variances: (list[float]) Variances of prior boxes.
        hw: (tuple[int, int]) height and width of the image.

    Return:
        decoded landm predictions
    """
    height, width = hw
    landms = torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
    scale = torch.tensor([width, height], device=landms.device).repeat(5)
    return landms * scale
