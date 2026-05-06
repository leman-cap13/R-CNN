from dataclasses import dataclass
from typing import Dict, List

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]

PENNFUDAN_CLASSES = ["person"]


def build_class_maps(class_names: List[str]):
    """
    Class index convention:
        0 = background
        1..N = object classes
    """
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    idx_to_class = {0: "background", **{i + 1: name for i, name in enumerate(class_names)}}
    return class_to_idx, idx_to_class


CLASS_TO_IDX, IDX_TO_CLASS = build_class_maps(VOC_CLASSES)
PENNFUDAN_CLASS_TO_IDX, PENNFUDAN_IDX_TO_CLASS = build_class_maps(PENNFUDAN_CLASSES)


@dataclass
class Config:
    # Default is now Penn-Fudan because it is easier to run quickly.
    dataset_type: str = "pennfudan"  # "pennfudan" or "voc"
    dataset_root: str = "PennFudanPed"
    train_split: str = "train"
    val_split: str = "val"

    image_height: int = 600
    image_width: int = 800
    roi_size: int = 224

    # For Penn-Fudan: background + person = 2.
    # For VOC this will be overridden in main.py.
    num_classes: int = len(PENNFUDAN_CLASSES) + 1

    max_proposals_train: int = 400
    max_proposals_test: int = 800

    samples_per_image: int = 64
    positive_fraction: float = 0.25
    positive_iou_threshold: float = 0.5
    negative_iou_threshold: float = 0.3

    batch_size: int = 1
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

    score_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections_per_image: int = 100

    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
