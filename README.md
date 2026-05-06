# Educational R-CNN Object Detector in PyTorch

This project is a from-scratch, educational R-CNN-style detector. It does **not** use `torchvision.models.detection`.

It supports two dataset formats:

1. **Penn-Fudan Pedestrian Dataset** — recommended for first run
2. **Pascal VOC-style XML annotations**

The Penn-Fudan version trains a 2-class detector:

```text
0 = background
1 = person
```

---

## Project structure

```text
educational_rcnn_project/
├── main.py                         # training entry point
├── predict.py                      # prediction / visualization entry point
├── requirements.txt
├── README.md
├── checkpoints/                    # saved model checkpoints
├── outputs/                        # prediction images
├── scripts/
│   ├── create_tiny_voc_example.py
│   └── download_penn_fudan.sh
└── src/
    └── rcnn_detector/
        ├── config.py               # class names and hyperparameters
        ├── dataset.py              # VOC + Penn-Fudan datasets
        ├── box_ops.py              # IoU, NMS, box encode/decode
        ├── proposals.py            # region proposals, sampling, ROI crop
        ├── model.py                # CNN backbone + heads
        ├── losses.py               # multi-task R-CNN loss
        ├── train.py                # training and validation loops
        ├── inference.py            # prediction pipeline
        └── visualize.py            # draw boxes on images
```

---

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Download Penn-Fudan

From the project root:

```bash
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -q PennFudanPed.zip
```

or:

```bash
bash scripts/download_penn_fudan.sh
```

Expected layout:

```text
PennFudanPed/
├── PNGImages/
│   ├── FudanPed00001.png
│   ├── FudanPed00002.png
│   └── ...
└── PedMasks/
    ├── FudanPed00001_mask.png
    ├── FudanPed00002_mask.png
    └── ...
```

The dataset class reads the mask files and converts each pedestrian instance into a bounding box.

---

## 3. Train on Penn-Fudan

```bash
python main.py --dataset-type pennfudan --dataset-root PennFudanPed --epochs 5
```

For a quick smoke test:

```bash
python main.py --dataset-type pennfudan --dataset-root PennFudanPed --epochs 1
```

Force CPU:

```bash
python main.py --dataset-type pennfudan --dataset-root PennFudanPed --epochs 1 --cpu
```

Checkpoints are saved to:

```text
checkpoints/rcnn_epoch_1.pth
checkpoints/rcnn_epoch_2.pth
...
```

---

## 4. Predict on one Penn-Fudan image

```bash
python predict.py \
  --image PennFudanPed/PNGImages/FudanPed00001.png \
  --checkpoint checkpoints/rcnn_epoch_5.pth \
  --output outputs/fudan_prediction.jpg
```

If you get no detections early in training, lower the threshold:

```bash
python predict.py \
  --image PennFudanPed/PNGImages/FudanPed00001.png \
  --checkpoint checkpoints/rcnn_epoch_1.pth \
  --output outputs/fudan_prediction.jpg \
  --score-threshold 0.1
```

The prediction image will be saved to:

```text
outputs/fudan_prediction.jpg
```

---

## 5. How Penn-Fudan labels are created

Penn-Fudan does not use Pascal VOC XML files. It uses instance masks.

For each mask:

```text
0     = background
1..N  = pedestrian object ids
255   = ignored boundary in some masks
```

The dataset code does this:

```text
mask instance id → pixel region → xmin, ymin, xmax, ymax → label person
```

So every valid object becomes:

```python
box = [xmin, ymin, xmax, ymax]
label = 1  # person
```

---

## 6. Pascal VOC mode

If you still want Pascal VOC-style XML annotations, use:

```bash
python main.py --dataset-type voc --dataset-root data/VOCdevkit/VOC2007 --epochs 5
```

Expected VOC layout:

```text
data/VOCdevkit/VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/
    └── Main/
        ├── train.txt
        └── val.txt
```

---

## Senior-engineer note

This is intentionally slow because it is educational R-CNN:

```text
proposal crop → CNN
proposal crop → CNN
proposal crop → CNN
```

Fast R-CNN and Faster R-CNN are faster because they run the CNN once on the full image and crop features instead of image pixels.


## Full workflow from zero

### Linux / macOS

```bash
git clone https://github.com/leman-cap13/R-CNN.git
cd R-CNN

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -q PennFudanPed.zip

python main.py --dataset-type pennfudan --dataset-root PennFudanPed --epochs 10

python predict.py \
  --image PennFudanPed/PNGImages/FudanPed00001.png \
  --checkpoint checkpoints/rcnn_epoch_10.pth \
  --output outputs/fudan_prediction.jpg \
  --score-threshold 0.1
```

---

### Windows PowerShell

```powershell
git clone https://github.com/leman-cap13/R-CNN.git
cd R-CNN

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

Invoke-WebRequest `
  -Uri "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip" `
  -OutFile "PennFudanPed.zip"

Expand-Archive -Path "PennFudanPed.zip" -DestinationPath "." -Force

python main.py --dataset-type pennfudan --dataset-root PennFudanPed --epochs 1

python predict.py `
  --image PennFudanPed/PNGImages/FudanPed00001.png `
  --checkpoint checkpoints/rcnn_epoch_1.pth `
  --output outputs/fudan_prediction.jpg `
  --score-threshold 0.1
```
