import os
import torch
from .proposals import crop_and_resize_rois, generate_region_proposals, label_and_sample_proposals
from .losses import rcnn_loss
from .inference import predict_single_image


def train_one_epoch(model, dataloader, optimizer, device, cfg, epoch: int) -> None:
    model.train()
    running_loss = 0.0
    for step, (images, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_loss = 0.0
        loss_dicts = []
        for image, target in zip(images, targets):
            image = image.to(device)
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            proposals = generate_region_proposals(image.detach().cpu(), cfg.max_proposals_train, use_selective_search=True).to(device)
            sampled_proposals, sampled_labels, bbox_targets = label_and_sample_proposals(
                proposals, gt_boxes, gt_labels,
                cfg.samples_per_image, cfg.positive_fraction,
                cfg.positive_iou_threshold, cfg.negative_iou_threshold,
            )
            roi_images = crop_and_resize_rois(image, sampled_proposals, cfg.roi_size)
            if roi_images.numel() == 0:
                continue
            class_logits, bbox_deltas = model(roi_images)
            loss, loss_dict = rcnn_loss(class_logits, bbox_deltas, sampled_labels, bbox_targets)
            batch_loss = batch_loss + loss
            loss_dicts.append(loss_dict)
        if isinstance(batch_loss, float):
            continue
        batch_loss.backward()
        optimizer.step()
        running_loss += float(batch_loss.detach().cpu())
        if step % 10 == 0:
            if loss_dicts:
                avg_total = sum(d["total_loss"] for d in loss_dicts) / len(loss_dicts)
                avg_cls = sum(d["cls_loss"] for d in loss_dicts) / len(loss_dicts)
                avg_reg = sum(d["reg_loss"] for d in loss_dicts) / len(loss_dicts)
            else:
                avg_total = avg_cls = avg_reg = 0.0
            print(f"Epoch [{epoch}] Step [{step}/{len(dataloader)}] Loss: {avg_total:.4f} Cls: {avg_cls:.4f} Reg: {avg_reg:.4f}")
    print(f"Epoch [{epoch}] average loss: {running_loss / max(1, len(dataloader)):.4f}")


@torch.no_grad()
def validate(model, dataloader, device, cfg, max_images: int = 20) -> None:
    model.eval()
    total_images, total_predictions = 0, 0
    for images, targets in dataloader:
        for image, target in zip(images, targets):
            pred = predict_single_image(model, image, device, cfg)
            total_images += 1
            total_predictions += pred["boxes"].shape[0]
            print(f"Validation image {target['image_id']}: {pred['boxes'].shape[0]} predictions")
            if total_images >= max_images:
                print(f"Average predictions per image: {total_predictions / max(1, total_images):.2f}")
                return
    print(f"Average predictions per image: {total_predictions / max(1, total_images):.2f}")


def save_checkpoint(model, optimizer, epoch: int, cfg, class_names=None, class_to_idx=None, idx_to_class=None) -> str:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"rcnn_epoch_{epoch}.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "dataset_type": getattr(cfg, "dataset_type", None),
            "num_classes": getattr(cfg, "num_classes", None),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")
    return path
