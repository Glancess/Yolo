import torch
import torch.nn as nn
from torchvision.ops import box_iou, generalized_box_iou

import torch.optim as optim


def xywh_to_xyxy(box):

    x = box[:, 0]
    y = box[:, 1]
    w = box[:, 2]
    h = box[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2

    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def assignment(pred_bbox, gt_bbox):

    pred_xyxy = xywh_to_xyxy(pred_bbox)
    gt_xyxy = xywh_to_xyxy(gt_bbox)

    ious = box_iou(pred_xyxy, gt_xyxy)

    best_iou, best_idx = ious.max(dim=0)

    # tensor([
    # 1,
    # 2,
    # 99
    # ])
    return best_idx  # 传出来GT1，2，3的最可能的三个预测框


#              GT1     GT2     GT3

# P1           0.1     0.2     0.3

# P2           0.8     0.1     0.2

# P3           0.2     0.9     0.1

# P4           0.1     0.1     0.2

# ...

# P100         0.3     0.1     0.4


class DetectionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.conf_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()

    # pred=16*100*9
    def forward(self, pred, targets):
        total_loss = 0  # 用于累加
        for i in range(len(targets)):

            # 当前图片的100个预测框

            pred_i = pred[i]
            pred_bbox = torch.sigmoid(pred_i[:, :4])  # 100 * 4
            pred_conf = pred_i[:, 4]  # 100 * 1
            pred_cls = pred_i[:, 5:]  # 100 * 4

            # 当前图片真实目标

            target_i = targets[i]

            gt_cls = target_i[:, 0].long()
            gt_bbox = target_i[:, 1:5]
            if target_i.shape[0] == 0:

                # 所有预测框都是背景
                conf_target = torch.zeros(pred_bbox.shape[0], device=pred.device)

                loss_conf = self.conf_loss(pred_conf, conf_target)

                total_loss += loss_conf

                continue
            else:
                best_idx = assignment(pred_bbox, gt_bbox)

                positive_pred = pred_bbox[
                    best_idx
                ]  # 取出来 最大可能的几个预测框，计算loss了。

                # 计算Giou
                pred_xyxy = xywh_to_xyxy(positive_pred)

                gt_xyxy = xywh_to_xyxy(gt_bbox)
                giou = generalized_box_iou(pred_xyxy, gt_xyxy)
                giou = torch.diag(giou)

                loss_box = (1 - giou).mean()
                # 计算Giou

                # confidence LOSS
                conf_target = torch.zeros(100, device=pred.device)
                conf_target[best_idx] = 1
                loss_conf = self.conf_loss(pred_conf, conf_target)
                # confidence LOSS
                # 分类loss
                positive_cls_pred = pred_cls[best_idx]

                loss_cls = self.cls_loss(positive_cls_pred, gt_cls)
                # 分类loss

                loss = 5 * loss_box + loss_conf + loss_cls
                total_loss += loss
        return total_loss / len(targets)


if __name__ == "__main__":

    pred = torch.randn(2, 100, 9, requires_grad=True)

    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.3, 0.4, 0.1, 0.1]]),
        torch.tensor([[2, 0.6, 0.6, 0.2, 0.2]]),
    ]

    criterion = DetectionLoss()

    loss = criterion(pred, targets)

    print("loss:", loss)

    loss.backward()

    print(pred.grad is None)
