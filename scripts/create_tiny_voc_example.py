"""Create a tiny fake VOC-like dataset for smoke testing.
This is not a real training dataset. It only checks that the project runs.
"""
import os
from PIL import Image, ImageDraw

root = "data/VOCdevkit/VOC2007"
os.makedirs(f"{root}/JPEGImages", exist_ok=True)
os.makedirs(f"{root}/Annotations", exist_ok=True)
os.makedirs(f"{root}/ImageSets/Main", exist_ok=True)

for image_id in ["000001", "000002"]:
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 80, 260, 220], outline="black", width=4)
    img.save(f"{root}/JPEGImages/{image_id}.jpg")
    xml = f"""<annotation>
    <folder>VOC2007</folder>
    <filename>{image_id}.jpg</filename>
    <size><width>400</width><height>300</height><depth>3</depth></size>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox><xmin>120</xmin><ymin>80</ymin><xmax>260</xmax><ymax>220</ymax></bndbox>
    </object>
</annotation>"""
    with open(f"{root}/Annotations/{image_id}.xml", "w", encoding="utf-8") as f:
        f.write(xml)

with open(f"{root}/ImageSets/Main/train.txt", "w", encoding="utf-8") as f:
    f.write("000001\n")
with open(f"{root}/ImageSets/Main/val.txt", "w", encoding="utf-8") as f:
    f.write("000002\n")
print("Tiny VOC example created.")
