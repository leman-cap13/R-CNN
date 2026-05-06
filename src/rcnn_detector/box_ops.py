from typing import Tuple
import torch


def xyxy_to_cxcywh(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def encode_boxes(proposals: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    px, py, pw, ph = xyxy_to_cxcywh(proposals)
    gx, gy, gw, gh = xyxy_to_cxcywh(gt_boxes)
    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    px, py, pw, ph = xyxy_to_cxcywh(proposals)
    tx, ty, tw, th = deltas.unbind(dim=1)
    tw = tw.clamp(min=-5, max=5)
    th = th.clamp(min=-5, max=5)
    gx = tx * pw + px
    gy = ty * ph + py
    gw = torch.exp(tw) * pw
    gh = torch.exp(th) * ph
    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, width - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, width - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, height - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, height - 1)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float = 8.0) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep.nonzero(as_tuple=False).squeeze(1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep)
