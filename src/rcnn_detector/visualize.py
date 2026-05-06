from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def visualize_predictions(image: torch.Tensor, prediction: Dict[str, torch.Tensor], save_path: str, idx_to_class: Dict[int, str]) -> None:
    pil_image = tensor_to_pil(image)
    draw = ImageDraw.Draw(pil_image)
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = None
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        class_name = idx_to_class.get(int(label), str(int(label)))
        text = f"{class_name}: {float(score):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), text, font=font) if font else draw.textbbox((x1, y1), text)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)
    pil_image.save(save_path)
    print(f"Saved visualization to: {save_path}")
