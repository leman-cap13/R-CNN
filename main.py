import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.rcnn_detector.config import (
    Config,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    VOC_CLASSES,
    PENNFUDAN_CLASSES,
    PENNFUDAN_CLASS_TO_IDX,
    PENNFUDAN_IDX_TO_CLASS,
)
from src.rcnn_detector.dataset import VOCDataset, PennFudanPedDataset, collate_fn
from src.rcnn_detector.model import RCNNDetector
from src.rcnn_detector.train import train_one_epoch, validate, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train educational R-CNN detector")
    parser.add_argument(
        "--dataset-type",
        choices=["pennfudan", "voc"],
        default="pennfudan",
        help="Dataset format. Use pennfudan for PennFudanPed/PNGImages + PedMasks.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="For Penn-Fudan: PennFudanPed. For VOC: data/VOCdevkit/VOC2007.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--score-threshold", type=float, default=None, help="Validation score threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def build_datasets(cfg: Config):
    image_size = (cfg.image_height, cfg.image_width)

    if cfg.dataset_type == "pennfudan":
        class_names = PENNFUDAN_CLASSES
        class_to_idx = PENNFUDAN_CLASS_TO_IDX
        idx_to_class = PENNFUDAN_IDX_TO_CLASS
        cfg.num_classes = len(class_names) + 1

        train_dataset = PennFudanPedDataset(
            root=cfg.dataset_root,
            split="train",
            image_size=image_size,
            class_to_idx=class_to_idx,
        )
        val_dataset = PennFudanPedDataset(
            root=cfg.dataset_root,
            split="val",
            image_size=image_size,
            class_to_idx=class_to_idx,
        )
    else:
        class_names = VOC_CLASSES
        class_to_idx = CLASS_TO_IDX
        idx_to_class = IDX_TO_CLASS
        cfg.num_classes = len(class_names) + 1

        train_dataset = VOCDataset(
            root=cfg.dataset_root,
            split=cfg.train_split,
            image_size=image_size,
            class_to_idx=class_to_idx,
        )
        val_dataset = VOCDataset(
            root=cfg.dataset_root,
            split=cfg.val_split,
            image_size=image_size,
            class_to_idx=class_to_idx,
        )

    return train_dataset, val_dataset, class_names, class_to_idx, idx_to_class


def main():
    args = parse_args()
    cfg = Config()
    cfg.dataset_type = args.dataset_type

    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root
    else:
        cfg.dataset_root = "PennFudanPed" if cfg.dataset_type == "pennfudan" else "data/VOCdevkit/VOC2007"

    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.score_threshold is not None:
        cfg.score_threshold = args.score_threshold

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset type: {cfg.dataset_type}")
    print(f"Dataset root: {cfg.dataset_root}")

    train_dataset, val_dataset, class_names, class_to_idx, idx_to_class = build_datasets(cfg)
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = RCNNDetector(num_classes=cfg.num_classes, feature_dim=512).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(1, cfg.num_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        validate(model, val_loader, device, cfg, max_images=5)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            class_names=class_names,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
        )


if __name__ == "__main__":
    main()
