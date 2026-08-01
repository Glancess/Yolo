import torch
import torch.nn as nn

class DetectionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.box_loss = nn.MSELoss()

        self.conf_loss = nn.BCEWithLogitsLoss()


    def forward(self,pred,target):

        bbox_pred = pred[:,:4]
        conf_pred = pred[:,4]


        bbox_target = target[:,:4]
        conf_target = target[:,4]


        # 所有图片计算confidence
        loss_conf = self.conf_loss(
            conf_pred,
            conf_target
        )


        # 只计算有目标图片的框
        mask = conf_target == 1


        if mask.sum()>0:

            loss_box = self.box_loss(
                bbox_pred[mask],
                bbox_target[mask]
            )

        else:

            loss_box = torch.tensor(
                0.0,
                device=pred.device
            )


        loss = loss_box + loss_conf


        return loss