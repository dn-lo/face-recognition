"""Face detection script.

This module implements a command-line interface to run face detection using a selected backbone.
It loads a trained model checkpoint, prepares input images, runs inference, decodes bounding boxes
and landmarks, applies non-maximum suppression (NMS), and saves sa visualization of detected faces.
"""

import argparse
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.backends import cudnn

from models.retinaface.config import CONFIG_MOBILE_NET, CONFIG_RESNET_50
from models.retinaface.decoder import Decoder
from models.retinaface.retinaface import RetinaFace
from utils.image import Color, draw_box, draw_landmarks, load_image
from utils.model import load_model
from utils.py_cpu_nms import py_cpu_nms


class Backbone(Enum):
    """Enumeration of supported backbone network types."""

    MOBILE_0_25 = "mobile_0_25"
    RESNET_50 = "resnet_50"


NET_TO_CONFIG = {
    Backbone.MOBILE_0_25: CONFIG_MOBILE_NET,
    Backbone.RESNET_50: CONFIG_RESNET_50,
}
# Checkpoints downloaded from: https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1
NET_TO_WEIGHTS = {
    Backbone.MOBILE_0_25: "mobilenet0.25_Final.pth",
    Backbone.RESNET_50: "Resnet50_Final.pth",
}
assert set(NET_TO_CONFIG.keys()) == set(NET_TO_WEIGHTS.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--checkpoint", default="", type=str, help="Checkpoint path")
    default_image = "https://wallpapercave.com/wp/wp6552551.jpg"  # To Pimp a Butterfly album cover
    parser.add_argument(
        "-i", "--input_image", default=default_image, type=str, help="Input image path or URL"
    )
    parser.add_argument("--network", default="resnet_50", type=Backbone, help="Network backbone")
    parser.add_argument("--cpu", action="store_false", default=True, help="Use cpu inference")
    parser.add_argument(
        "--confidence_threshold", default=0.02, type=float, help="Confidence threshold"
    )
    parser.add_argument("--top_k", default=5000, type=int, help="Keep top K boxes before NMS")
    parser.add_argument("--nms_threshold", default=0.4, type=float, help="NMS threshold")
    parser.add_argument("--nms_top_k", default=750, type=int, help="Keep top K boxes after NMS")
    parser.add_argument("--view_threshold", default=0.6, type=float, help="Visualization threshold")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = NET_TO_CONFIG.get(Backbone(args.network), None)

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    device = torch.device("cpu" if args.cpu else "cuda")
    default_checkpoint = Path.cwd() / "weights/retinaface" / NET_TO_WEIGHTS.get(args.network, None)
    checkpoint = args.checkpoint or default_checkpoint
    net = load_model(net, checkpoint, device)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)

    bgr_mean_imagenet = (104, 117, 123)  # Mean BGR values in ImageNet, used for training
    img, img_raw, downscale = load_image(
        args.input_image, bgr_mean=bgr_mean_imagenet, max_size_mp=4
    )
    img = img.to(device)
    image_size = img.shape[-2:]

    tic = time.time()
    locations, confidences, landmarks = net(img)  # forward pass
    print(f"net forward time: {time.time() - tic:.4f} s")

    decoder = Decoder(cfg, image_size, device=device, downscale=downscale)
    boxes = decoder.decode_boxes(locations.data.squeeze(0)).cpu().numpy()
    landmarks = decoder.decode_landmarks(landmarks.data.squeeze(0)).cpu().numpy()
    landmarks = landmarks.astype(np.float32, copy=False)
    scores = confidences.squeeze(0).data.cpu().numpy()[:, 1].astype(np.float32, copy=False)

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
    keep = keep[: args.nms_top_k]
    boxes = boxes[keep]
    scores = scores[keep]
    landmarks = landmarks[keep]

    # show image with detections
    for box, score, landmark in zip(boxes, scores, landmarks, strict=False):
        if score < args.view_threshold:
            continue

        draw_box(img_raw, box, color=Color.RED)
        cx = int(box[0])
        cy = int(box[1] + 12)
        text = f"{score:.4f}"
        cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, Color.WHITE.value)

        draw_landmarks(img_raw, landmark)

    output_file = Path.cwd() / "outputs" / Path(args.input_image).name.replace(".", "_detected.")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    cv2.imwrite(output_file, img_raw)
    print(f"Output saved to {output_file}")
