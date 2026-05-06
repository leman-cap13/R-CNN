import os
import argparse
import numpy as np
from PIL import Image
import torch

from src.rcnn_detector.config import (
    Config,
    IDX_TO_CLASS,
    PENNFUDAN_IDX_TO_CLASS,
    VOC_CLASSES,
    PENNFUDAN_CLASSES,
)
from src.rcnn_detector.model import RCNNDetector
from src.rcnn_detector.inference import predict_single_image
from src.rcnn_detector.visualize import visualize_predictions


def load_image_as_tensor(image_path: str, cfg: Config) -> torch.Tensor:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((cfg.image_width, cfg.image_height), resample=Image.BILINEAR)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction using trained educational R-CNN")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="outputs/prediction.jpg", help="Where to save visualized result")
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "pennfudan", "voc"],
        default="auto",
        help="Usually keep auto. Used only if the checkpoint lacks metadata.",
    )
    parser.add_argument("--score-threshold", type=float, default=None, help="Override score threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def class_info_from_checkpoint(checkpoint, dataset_type_arg: str):
    if checkpoint.get("class_names") is not None and checkpoint.get("idx_to_class") is not None:
        idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
        return int(checkpoint["num_classes"]), idx_to_class

    # Fallback for old checkpoints.
    dataset_type = checkpoint.get("dataset_type") or dataset_type_arg
    if dataset_type == "auto":
        dataset_type = "pennfudan"

    if dataset_type == "pennfudan":
        return len(PENNFUDAN_CLASSES) + 1, PENNFUDAN_IDX_TO_CLASS
    return len(VOC_CLASSES) + 1, IDX_TO_CLASS


def main():
    args = parse_args()
    cfg = Config()
    if args.score_threshold is not None:
        cfg.score_threshold = args.score_threshold

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes, idx_to_class = class_info_from_checkpoint(checkpoint, args.dataset_type)
    cfg.num_classes = num_classes

    model = RCNNDetector(num_classes=num_classes, feature_dim=512).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_image_as_tensor(args.image, cfg)
    prediction = predict_single_image(model, image, device, cfg)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    visualize_predictions(image, prediction, args.output, idx_to_class)

    print("Predictions:")
    if prediction["boxes"].numel() == 0:
        print("No detections. Try lowering --score-threshold, e.g. --score-threshold 0.1")
        return

    for box, label, score in zip(
        prediction["boxes"].cpu(),
        prediction["labels"].cpu(),
        prediction["scores"].cpu(),
    ):
        print({
            "class": idx_to_class.get(int(label), str(int(label))),
            "score": round(float(score), 4),
            "box_xyxy": [round(float(v), 2) for v in box.tolist()],
        })


if __name__ == "__main__":
    main()
