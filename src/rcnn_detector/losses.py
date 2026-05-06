from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def rcnn_loss(class_logits: torch.Tensor, bbox_deltas: torch.Tensor, labels: torch.Tensor, bbox_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls_loss = F.cross_entropy(class_logits, labels)
    positive_mask = labels > 0
    if positive_mask.any():
        reg_loss = F.smooth_l1_loss(bbox_deltas[positive_mask], bbox_targets[positive_mask], reduction="mean")
    else:
        reg_loss = bbox_deltas.sum() * 0.0
    total = cls_loss + reg_loss
    return total, {
        "total_loss": float(total.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
    }
