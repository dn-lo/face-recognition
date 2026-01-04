"""Image utils (basic CV loading, operations and preprocessing)."""

import math
from enum import Enum
from urllib.request import urlopen

import cv2
import numpy as np
import torch


def load_image(
    image_path: str,
    bgr_mean: tuple[int, int, int] | None = None,
    max_size_mp: int | None = None,
) -> tuple[torch.Tensor, np.ndarray, float]:
    """Load and preprocess an image, resizing if required to fit within a max size limit.

    Args:
        image_path: Path or URL pointing to input image.
        bgr_mean: Mean values for BGR channels for normalization. Defaults to None.
        max_size_mp: Optional max allowed image size in megapixel. If exceeded, the image
            will be resized to fit within the limit while maintaining aspect ratio.

    Returns:
        tensor: Preprocessed image tensor of shape (1, 3, H_resized, W_resized).
        img_raw: original BGR image as read by OpenCV, of shape (H, W, 3).
        resize_scale: float factor to map coordinates from the preprocessed image back to
            the original image (original_dim / preprocessed_dim). 1.0 if no resizing occurred.
    """
    # URL
    if image_path.startswith(("http://", "https://")):
        try:
            with urlopen(image_path) as response:  # noqa: S310
                image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            msg = f"Could not download image from URL: {image_path}. Error: {e}"
            raise RuntimeError(msg) from e
    # Local file
    else:
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_raw is None:
        msg = f"Could not decode image from {image_path}"
        raise FileNotFoundError(msg)

    orig_h, orig_w = img_raw.shape[:2]
    input_size_mp = orig_h * orig_w / 1e6
    print(f"Original image size: ({orig_w}x{orig_h}), {input_size_mp:.2f} megapixels.")

    if max_size_mp and input_size_mp > max_size_mp:
        # compute downscale factor (power of two) to reduce aliasing artifacts
        downscale = math.sqrt(input_size_mp / max_size_mp)
        downscale = 2 ** math.ceil(math.log2(downscale))

        if orig_h % downscale != 0 or orig_w % downscale != 0:
            orig_h -= orig_h % downscale
            orig_w -= orig_w % downscale
            img_raw = img_raw[:orig_h, :orig_w]
            print(f"Cropping image to ({orig_w}x{orig_h}) to be divisible by {downscale=}.")

        new_w = max(1, orig_w // downscale)
        new_h = max(1, orig_h // downscale)
        print(f"Resizing to ({new_w}x{new_h}) ({downscale=}) to fit {max_size_mp} megapixels.")
        img_resized = cv2.resize(img_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_raw
        downscale = 1.0

    img = np.float32(img_resized)
    if bgr_mean is not None:
        img -= bgr_mean

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    return img, img_raw, downscale


class Color(Enum):
    """Common colors in BGR format."""

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


def draw_box(img: np.ndarray, box: np.ndarray, color: Color) -> None:
    """Draw a bounding box on an image.

    Args:
        img: Image array in BGR format.
        box: Bounding box specified as (x_min, y_min, x_max, y_max).
        color: Box color.
    """
    x_min, y_min, x_max, y_max = box.astype(int).tolist()
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color.value, thickness=2)


def draw_landmarks(img: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw landmarks on an image.

    Args:
        img: Image array in BGR format.
        landmarks: Landmarks as a numpy array of shape (2*N,).
        color: Landmark color.
    """
    colors = list(Color)
    for i in range(0, len(landmarks), 2):
        x = int(landmarks[i])
        y = int(landmarks[i + 1])
        color = colors[(i // 2) % len(colors)]  # Cycle through colors if more landmarks than colors
        cv2.circle(img, (x, y), radius=1, color=color.value, thickness=4)
