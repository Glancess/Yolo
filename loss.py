import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou
def xywh_to_xyxy(box):

    x = box[:,0]
    y = box[:,1]
    w = box[:,2]
    h = box[:,3]


    x1 = x - w / 2
    y1 = y - h / 2

    x2 = x + w / 2
    y2 = y + h / 2


    return torch.stack(
        [x1,y1,x2,y2],
        dim=1
    )

class DetectionLoss(nn.Module):

    def __init__(self):
        super().__init__()


        self.conf_loss = nn.BCEWithLogitsLoss()


    def forward(self,pred,target):

        conf_pred = pred[:,4]
        conf_target = target[:,4]

        # 所有图片计算confidence
        loss_conf = self.conf_loss(
            conf_pred,
            conf_target
        )

        # 只计算有目标图片的框
        mask = conf_target == 1
        if mask.sum()>0:
            pred_bbox = torch.sigmoid(pred[mask, :4])

            target_bbox = target[mask,:4]

            pred_xyxy = xywh_to_xyxy(
                                pred_bbox
                            )
            target_xyxy = xywh_to_xyxy(
                                target_bbox
                            )
            
            iou = torch.diag(
            generalized_box_iou(
                 pred_xyxy,
                 target_xyxy
                )
                            )
            loss_box = 1 - iou.mean()
            # print("loss_box:", loss_box)
            # print("requires_grad:", loss_box.requires_grad)

        else:

            loss_box = torch.tensor(
                0.0,
                device=pred.device
            )


        loss = 3*loss_box + loss_conf


        return loss

