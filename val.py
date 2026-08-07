import torch
import torchvision

from torch.utils.data import DataLoader
from dataset import YoloDataset
from loss import xywh_to_xyxy, assignment
from torchvision.ops import box_iou

# =====================
# collate_fn
# =====================


def collate_fn(batch):

    imgs = []
    targets = []

    for img, target in batch:

        imgs.append(img)
        targets.append(target)

    imgs = torch.stack(imgs)

    return imgs, targets


# =====================
# 构造验证集
# =====================


def get_val_loader():

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]
    )

    val_dataset = YoloDataset(
        "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/images",
        "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/labels",
        transform,
        None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return val_loader


# =====================
# 验证函数
# =====================


def validate(model, device):

    val_loader = get_val_loader()

    model.eval()

    total_iou = 0
    count = 0

    iou50_success = 0

    with torch.no_grad():

        for imgs, targets in val_loader:

            imgs = imgs.to(device)

            targets = [t.to(device) for t in targets]

            pred = model(imgs)

            pred = pred.reshape(pred.size(0), 100, 9)

            # 遍历batch里面每张图片

            for i in range(len(targets)):

                target_i = targets[i]

                # 当前图片无目标

                if target_i.shape[0] == 0:
                    continue

                # --------------------
                # 预测框
                # --------------------

                pred_i = pred[i]

                pred_bbox = torch.sigmoid(pred_i[:, :4])

                # --------------------
                # GT
                # --------------------

                gt_bbox = target_i[:, 1:5]

                # --------------------
                # assignment
                # --------------------

                best_idx = assignment(pred_bbox, gt_bbox)

                # 取匹配成功的预测框

                positive_pred = pred_bbox[best_idx]

                # --------------------
                # IoU
                # --------------------

                pred_xyxy = xywh_to_xyxy(positive_pred)

                gt_xyxy = xywh_to_xyxy(gt_bbox)

                ious = box_iou(pred_xyxy, gt_xyxy)

                # 取对应位置

                iou = torch.diag(ious)

                total_iou += iou.sum().item()

                count += len(iou)

                # IoU@0.5

                iou50_success += (iou >= 0.5).sum().item()

        if count > 0:

            mean_iou = total_iou / count

            iou50_rate = iou50_success / count

        else:

            mean_iou = 0

            iou50_rate = 0

    return {"mean_iou": mean_iou, "iou50": iou50_rate, "num": count}
