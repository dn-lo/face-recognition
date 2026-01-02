"""Face lmark script.

This module implements a command-line interface to run face lmark using a selected backbone.
It loads a trained model checkpoint, prepares input images, runs inference, decodes bounding boxes
and landmarks, applies non-maximum suppression (NMS), and saves
a visualization of detected faces.
"""

import argparse
import time

import cv2
import numpy as np
import torch
from torch.backends import cudnn

from models.retinaface.box_utils import decode, decode_landmarks
from models.retinaface.config import CONFIG_MOBILE_NET, CONFIG_RESNET_50
from models.retinaface.layers import PriorBox
from models.retinaface.retinaface import RetinaFace
from utils.load_model import load_model
from utils.py_cpu_nms import py_cpu_nms


def load_image(
    image_path: str, bgr_mean: tuple[int, int, int] | None = None
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image.

    Args:
        image_path: Path to the input image.
        bgr_mean: Mean values for BGR channels for normalization. Defaults to None.

    Returns:
        torch.Tensor: Preprocessed image tensor, with shape (1, C, H, W).
        np.ndarray: Original image read from the path.
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    if bgr_mean is not None:
        img -= bgr_mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    return img, img_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-m",
        "--trained_model",
        default="./weights/retinaface/Resnet50_Final.pth",
        type=str,
        help="Trained state_dict file path to open",
    )
    parser.add_argument(
        "--network", default="resnet50", help="Backbone network mobile0.25 or resnet50"
    )
    parser.add_argument("--cpu", action="store_false", default=True, help="Use cpu inference")
    parser.add_argument(
        "--confidence_threshold", default=0.02, type=float, help="confidence_threshold"
    )
    parser.add_argument("--top_k", default=5000, type=int, help="Kept top K boxes before NMS")
    parser.add_argument("--nms_threshold", default=0.4, type=float, help="nms_threshold")
    parser.add_argument("--keep_top_k", default=750, type=int, help="Kept top K boxes after NMS")
    parser.add_argument(
        "-s", "--save_image", action="store_true", default=True, help="show lmark results"
    )
    parser.add_argument("--view_threshold", default=0.6, type=float, help="visualization threshold")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    network_to_config = {"mobile0.25": CONFIG_MOBILE_NET, "resnet50": CONFIG_RESNET_50}
    cfg = network_to_config.get(args.network, None)

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    device = torch.device("cpu" if args.cpu else "cuda")
    net = load_model(net, args.trained_model, device)
    net.eval()
    print(net)
    cudnn.benchmark = True
    net = net.to(device)

    resize = 1

    # testing begin
    for _ in range(1):
        image_path = "/workspaces/face-recognition/data/IMG_3504.jpg"
        bgr_mean_imagenet = (104, 117, 123)  # Mean BGR values in ImageNet, used for training
        img, img_raw = load_image(image_path, bgr_mean=bgr_mean_imagenet)
        img = img.to(device)

        height_width = img.shape[-2:]
        tic = time.time()
        locations, confidences, landmarks = net(img)  # forward pass
        print(f"net forward time: {time.time() - tic:.4f}")

        prior_box = PriorBox(cfg, image_size=height_width)
        priors = prior_box.forward()
        priors = priors.to(device)
        prior_data = priors.data

        boxes = decode(locations.data.squeeze(0), prior_data, cfg.variance, height_width)
        boxes = boxes / resize
        boxes = boxes.cpu().numpy()

        scores = confidences.squeeze(0).data.cpu().numpy()[:, 1].astype(np.float32, copy=False)

        landmarks = decode_landmarks(
            landmarks.data.squeeze(0), prior_data, cfg.variance, height_width
        )
        landmarks = landmarks / resize
        landmarks = landmarks.cpu().numpy().astype(np.float32, copy=False)

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: args.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS and keep top-K after NMS
        keep = py_cpu_nms(boxes, scores, args.nms_threshold)
        keep = keep[: args.keep_top_k]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # show image with detections
        if args.save_image:
            for box, score, landmark in zip(boxes, scores, landmarks, strict=False):
                if score < args.view_threshold:
                    continue

                text = f"{score:.4f}"
                bbox = box.astype(int).tolist()
                lmark = landmark.astype(int).tolist()

                cv2.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cx = bbox[0]
                cy = bbox[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landmarks
                cv2.circle(img_raw, (lmark[0], lmark[1]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (lmark[2], lmark[3]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (lmark[4], lmark[5]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (lmark[6], lmark[7]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (lmark[8], lmark[9]), 1, (255, 0, 0), 4)

            name = "test.jpg"
            cv2.imwrite(name, img_raw)
