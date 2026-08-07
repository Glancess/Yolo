from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision

from dataset import YoloDataset

# =========================
# 多目标检测 collate_fn
# =========================


def collate_fn(batch):

    imgs = []
    targets = []

    for img, target in batch:
        imgs.append(img)
        targets.append(target)

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


# =========================
# 路径
# =========================

val_image_path = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/images"
val_label_path = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/labels"


# =========================
# transform
# =========================

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]
)


# =========================
# Dataset
# =========================

val_dataset = YoloDataset(val_image_path, val_label_path, transform, None)


print("验证集数量:", len(val_dataset))


# =========================
# DataLoader
# =========================

val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=True, drop_last=False, collate_fn=collate_fn
)


# =========================
# 测试一个batch
# =========================

for imgs, targets in val_loader:

    print("======================")

    print("图片batch shape:", imgs.shape)

    print("target数量:", len(targets))

    for i, target in enumerate(targets):

        print(f"第{i}张图片 target:", target.shape)

        print(target)

    break
