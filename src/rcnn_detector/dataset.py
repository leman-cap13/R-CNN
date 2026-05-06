import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import CLASS_TO_IDX, PENNFUDAN_CLASS_TO_IDX


class VOCDataset(Dataset):
    """Pascal VOC-style object detection dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (600, 800),
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        self._validate_paths()

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip().split()[0] for line in f.readlines() if line.strip()]
        if not self.image_ids:
            raise ValueError(f"No image ids found in split file: {self.split_file}")

    def _validate_paths(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"JPEGImages folder not found: {self.image_dir}")
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f"Annotations folder not found: {self.annotation_dir}")
        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def parse_annotation(self, annotation_path: str):
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"Annotation file missing: {annotation_path}")
        try:
            tree = ET.parse(annotation_path)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML annotation: {annotation_path}. Error: {e}")

        root = tree.getroot()
        boxes, labels = [], []

        for obj in root.findall("object"):
            name_node = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_node is None or bndbox is None:
                continue

            class_name = name_node.text
            if class_name not in self.class_to_idx:
                continue

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except Exception as e:
                raise ValueError(f"Invalid bounding box in {annotation_path}: {e}")

            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"Invalid box with non-positive size in {annotation_path}: "
                    f"{xmin}, {ymin}, {xmax}, {ymax}"
                )

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])

        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size
        boxes, labels = self.parse_annotation(annotation_path)

        image_tensor, boxes = resize_image_and_boxes(image, boxes, self.image_size)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": image_id,
            "original_size": (original_h, original_w),
            "resized_size": self.image_size,
        }
        return image_tensor, target


class PennFudanPedDataset(Dataset):
    """
    Penn-Fudan Pedestrian dataset.

    Expected structure after:
        wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
        unzip -q PennFudanPed.zip

    PennFudanPed/
    ├── PNGImages/
    │   ├── FudanPed00001.png
    │   └── ...
    └── PedMasks/
        ├── FudanPed00001_mask.png
        └── ...

    The mask has one integer id per object. We convert each object mask into one box.
    Labels are always 1 = person.
    """

    def __init__(
        self,
        root: str = "PennFudanPed",
        split: str = "train",
        image_size: Tuple[int, int] = (600, 800),
        train_ratio: float = 0.8,
        seed: int = 42,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.seed = seed
        self.class_to_idx = class_to_idx or PENNFUDAN_CLASS_TO_IDX

        self.image_dir = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")
        self._validate_paths()

        all_images = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(".png")])
        if not all_images:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        # Deterministic split. Penn-Fudan is small, so this is good enough for education.
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_images))
        rng.shuffle(indices)

        split_at = int(len(all_images) * train_ratio)
        if split == "train":
            selected = indices[:split_at]
        elif split in {"val", "valid", "validation", "test"}:
            selected = indices[split_at:]
        else:
            raise ValueError("PennFudan split must be 'train' or 'val'")

        self.image_files = [all_images[i] for i in selected]
        if not self.image_files:
            raise ValueError(f"No images selected for split={split}. Check train_ratio={train_ratio}")

    def _validate_paths(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Penn-Fudan root not found: {self.root}\n"
                "Download it with:\n"
                "  wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip\n"
                "  unzip -q PennFudanPed.zip"
            )
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"PNGImages folder not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"PedMasks folder not found: {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _mask_path_from_image_file(self, image_file: str) -> str:
        stem = os.path.splitext(image_file)[0]
        return os.path.join(self.mask_dir, f"{stem}_mask.png")

    def _boxes_from_mask(self, mask_path: str):
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file missing: {mask_path}")

        mask = np.asarray(Image.open(mask_path))
        if mask.ndim != 2:
            raise ValueError(f"Expected a single-channel mask, got shape {mask.shape}: {mask_path}")

        obj_ids = np.unique(mask)
        # 0 = background. 255 is often boundary/ignore in segmentation masks.
        obj_ids = obj_ids[(obj_ids != 0) & (obj_ids != 255)]

        boxes = []
        for obj_id in obj_ids:
            ys, xs = np.where(mask == obj_id)
            if len(xs) == 0 or len(ys) == 0:
                continue
            xmin = float(xs.min())
            xmax = float(xs.max())
            ymin = float(ys.min())
            ymax = float(ys.max())
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            raise ValueError(f"No valid pedestrian instances found in mask: {mask_path}")

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.full((boxes.shape[0],), self.class_to_idx["person"], dtype=np.int64)
        return boxes, labels

    def __getitem__(self, idx: int):
        image_file = self.image_files[idx]
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = self._mask_path_from_image_file(image_file)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size
        boxes, labels = self._boxes_from_mask(mask_path)

        image_tensor, boxes = resize_image_and_boxes(image, boxes, self.image_size)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": image_id,
            "original_size": (original_h, original_w),
            "resized_size": self.image_size,
        }
        return image_tensor, target


def resize_image_and_boxes(image: Image.Image, boxes: np.ndarray, image_size: Tuple[int, int]):
    """Resize PIL image and scale xyxy boxes accordingly."""
    original_w, original_h = image.size
    new_h, new_w = image_size

    image = image.resize((new_w, new_h), resample=Image.BILINEAR)

    boxes = boxes.copy()
    if boxes.shape[0] > 0:
        boxes[:, [0, 2]] *= new_w / original_w
        boxes[:, [1, 3]] *= new_h / original_h

    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor, boxes


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
