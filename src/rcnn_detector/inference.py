from typing import Dict
import torch
import torch.nn.functional as F
from .box_ops import clip_boxes_to_image, decode_boxes, nms
from .proposals import crop_and_resize_rois, generate_region_proposals


@torch.no_grad()
def predict_single_image(model, image: torch.Tensor, device: torch.device, cfg) -> Dict[str, torch.Tensor]:
    model.eval()
    image = image.to(device)
    height, width = image.shape[-2:]
    proposals = generate_region_proposals(image.detach().cpu(), max_proposals=cfg.max_proposals_test, use_selective_search=True).to(device)
    if proposals.numel() == 0:
        return _empty_prediction(device)
    roi_images = crop_and_resize_rois(image, proposals, cfg.roi_size)
    if roi_images.numel() == 0:
        return _empty_prediction(device)
    class_logits, bbox_deltas = model(roi_images)
    probs = F.softmax(class_logits, dim=1)
    scores, labels = probs.max(dim=1)
    keep = (labels > 0) & (scores >= cfg.score_threshold)
    proposals, bbox_deltas, scores, labels = proposals[keep], bbox_deltas[keep], scores[keep], labels[keep]
    if proposals.numel() == 0:
        return _empty_prediction(device)
    boxes = clip_boxes_to_image(decode_boxes(proposals, bbox_deltas), height, width)
    final_boxes, final_scores, final_labels = [], [], []
    for class_id in labels.unique():
        mask = labels == class_id
        keep_idx = nms(boxes[mask], scores[mask], cfg.nms_threshold)
        final_boxes.append(boxes[mask][keep_idx])
        final_scores.append(scores[mask][keep_idx])
        final_labels.append(labels[mask][keep_idx])
    if not final_boxes:
        return _empty_prediction(device)
    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_labels = torch.cat(final_labels, dim=0)
    order = final_scores.argsort(descending=True)[:cfg.max_detections_per_image]
    return {"boxes": final_boxes[order], "labels": final_labels[order], "scores": final_scores[order]}


def _empty_prediction(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), device=device),
        "labels": torch.empty((0,), dtype=torch.long, device=device),
        "scores": torch.empty((0,), device=device),
    }
