import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .box_ops import box_iou, clip_boxes_to_image, encode_boxes, remove_small_boxes


def generate_grid_proposals(height: int, width: int, scales: List[int] = [64, 96, 128, 192, 256, 384], aspect_ratios: List[float] = [0.5, 1.0, 2.0], stride: int = 64, max_proposals: int = 1000) -> torch.Tensor:
    boxes = []
    for scale in scales:
        area = scale * scale
        for ratio in aspect_ratios:
            box_w = int(round(math.sqrt(area / ratio)))
            box_h = int(round(box_w * ratio))
            for cy in range(stride // 2, height, stride):
                for cx in range(stride // 2, width, stride):
                    boxes.append([cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    boxes = clip_boxes_to_image(boxes, height, width)
    boxes = boxes[remove_small_boxes(boxes, min_size=16)]
    boxes = torch.unique(boxes.round(), dim=0)
    if boxes.shape[0] > max_proposals:
        boxes = boxes[torch.randperm(boxes.shape[0])[:max_proposals]]
    return boxes


def generate_selective_search_proposals(image_tensor: torch.Tensor, max_proposals: int = 1000) -> Optional[torch.Tensor]:
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        return None
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_np)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    boxes = []
    for x, y, w, h in rects[:max_proposals * 2]:
        if w >= 16 and h >= 16:
            boxes.append([x, y, x + w, y + h])
    if not boxes:
        return None
    height, width = image_tensor.shape[-2:]
    boxes = torch.tensor(boxes, dtype=torch.float32)
    boxes = clip_boxes_to_image(boxes, height, width)
    boxes = boxes[remove_small_boxes(boxes, min_size=16)]
    boxes = torch.unique(boxes.round(), dim=0)
    return boxes[:max_proposals]


def generate_region_proposals(image_tensor: torch.Tensor, max_proposals: int, use_selective_search: bool = True) -> torch.Tensor:
    height, width = image_tensor.shape[-2:]
    proposals = generate_selective_search_proposals(image_tensor, max_proposals) if use_selective_search else None
    if proposals is None:
        proposals = generate_grid_proposals(height, width, max_proposals=max_proposals)
    return proposals


def label_and_sample_proposals(proposals: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, samples_per_image: int, positive_fraction: float, positive_iou_threshold: float, negative_iou_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = proposals.device
    if gt_boxes.numel() == 0:
        num_samples = min(samples_per_image, proposals.shape[0])
        sampled_idx = torch.randperm(proposals.shape[0], device=device)[:num_samples]
        return proposals[sampled_idx], torch.zeros((num_samples,), dtype=torch.long, device=device), torch.zeros((num_samples, 4), dtype=torch.float32, device=device)
    proposals = torch.cat([proposals, gt_boxes], dim=0)
    ious = box_iou(proposals, gt_boxes)
    max_ious, matched_gt_indices = ious.max(dim=1)
    positive_indices = torch.where(max_ious >= positive_iou_threshold)[0]
    negative_indices = torch.where(max_ious < negative_iou_threshold)[0]
    num_pos = min(int(samples_per_image * positive_fraction), positive_indices.numel())
    num_neg = min(samples_per_image - num_pos, negative_indices.numel())
    positive_indices = positive_indices[torch.randperm(positive_indices.numel(), device=device)[:num_pos]] if positive_indices.numel() else positive_indices
    negative_indices = negative_indices[torch.randperm(negative_indices.numel(), device=device)[:num_neg]] if negative_indices.numel() else negative_indices
    sampled_indices = torch.cat([positive_indices, negative_indices], dim=0)
    if sampled_indices.numel() == 0:
        sampled_indices = torch.arange(min(samples_per_image, proposals.shape[0]), device=device)
        num_pos = 0
    sampled_proposals = proposals[sampled_indices]
    matched_indices = matched_gt_indices[sampled_indices]
    sampled_labels = gt_labels[matched_indices]
    sampled_labels[num_pos:] = 0
    bbox_targets = torch.zeros((sampled_indices.numel(), 4), dtype=torch.float32, device=device)
    if num_pos > 0:
        bbox_targets[:num_pos] = encode_boxes(sampled_proposals[:num_pos], gt_boxes[matched_indices[:num_pos]])
    return sampled_proposals, sampled_labels, bbox_targets


def crop_and_resize_rois(image: torch.Tensor, boxes: torch.Tensor, output_size: int) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(f"Expected image Tensor[3,H,W], got shape {image.shape}")
    device = image.device
    _, height, width = image.shape
    if boxes.numel() == 0:
        return torch.empty((0, 3, output_size, output_size), device=device)
    boxes = clip_boxes_to_image(boxes, height, width)
    rois = []
    for box in boxes:
        x1, y1, x2, y2 = box.round().long().tolist()
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 <= x1:
            x2 = min(x1 + 1, width - 1)
        if y2 <= y1:
            y2 = min(y1 + 1, height - 1)
        crop = image[:, y1:y2 + 1, x1:x2 + 1].unsqueeze(0)
        rois.append(F.interpolate(crop, size=(output_size, output_size), mode="bilinear", align_corners=False))
    return torch.cat(rois, dim=0)
