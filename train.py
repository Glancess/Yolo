
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import YoloDataset
from model import MymoduleforYolo
from loss import DetectionLoss
from torchvision.ops import generalized_box_iou
from torchvision.ops import box_iou
import torch.optim as optim


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










transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512,512)),
    torchvision.transforms.ToTensor()
])
train_dataset=YoloDataset("/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_train/images","/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_train/labels" ,transform,None)
MotorDataloader=DataLoader(train_dataset,batch_size=32,drop_last=False,shuffle=True)

val_dataset = YoloDataset(

    "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/images",

    "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/labels",

    transform,

    None

)

ValDataloader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    drop_last=False

)


mymoudle=MymoduleforYolo()#模型
criterion=DetectionLoss()#损失


for param in mymoudle.img_contract.parameters():
    param.requires_grad = False

optimizer = optim.Adam(

    mymoudle.parameters(),

    lr=1e-4

)#优化器

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymoudle.to(device)

best_loss = float("inf")

for epoch in range(30):
    print(f"--------开始第 {epoch} 轮训练----------")
    mymoudle.train()
    
    loss_sum=0
    step=1
    for img,target in MotorDataloader:

        img = img.to(device)
        target = target.to(device)

        pred = mymoudle(img)

        loss = criterion(pred,target)
        #print(f"第{step}步的loss为{loss}")
        step+=1;

        loss_sum+=loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    #print(f"--------第 {epoch} 轮 LOSS {loss_sum}----------")


    print(f"--------开始第 {epoch} 轮验证----------")

    best_iou=-1

    mymoudle.eval()
    val_loss_sum = 0
    total_iou = 0
    count = 0

    pos_conf_sum = 0
    neg_conf_sum = 0
    pos_count = 0
    neg_count = 0


    pos_correct = 0
    neg_correct = 0
    pos_total = 0
    neg_total = 0

    iou50_success=0

    with torch.no_grad():

        for img, target in ValDataloader:

            img = img.to(device)
            target = target.to(device)
            pred = mymoudle(img)
            loss = criterion(
                pred,
                target
            )

            val_loss_sum += loss.item()


            conf = torch.sigmoid(pred[:,4])#将预测转换为概率


            # --------------------
            # 置信度计算
            # --------------------
            target_conf = target[:,4]
            pos_mask = target_conf == 1
            if pos_mask.sum()>0:
                pos_conf_sum += conf[pos_mask].sum().item()
                pos_count += pos_mask.sum().item()

            neg_mask = target_conf == 0

            if neg_mask.sum()>0:

                neg_conf_sum += conf[neg_mask].sum().item()

                neg_count += neg_mask.sum().item()
            # --------------------
            # 置信度计算
            # --------------------




            # --------------------
            # 正确率计算
            # --------------------
            pred_label = conf > 0.5

            if pos_mask.sum()>0:
                pos_correct += (
                    pred_label[pos_mask]
                ).sum().item()

                pos_total += pos_mask.sum().item()

            if neg_mask.sum()>0:

                neg_correct += (
                    (~pred_label[neg_mask])
                ).sum().item()

                neg_total += neg_mask.sum().item()
            # --------------------
            # 正确率计算
            # --------------------


            
            # --------------------
            # IoU计算（越大越好，重合程度高）
            # --------------------

            mask = target[:,4] == 1


            if mask.sum() > 0:

                pred_bbox = torch.sigmoid(
                    pred[mask,:4]
                    )
            
                target_bbox = target[mask,:4]


                pred_xyxy = xywh_to_xyxy(
                    pred_bbox
                )

                target_xyxy = xywh_to_xyxy(
                    target_bbox
                )


                ious = box_iou(
                    pred_xyxy,
                    target_xyxy
                )


                iou = torch.diag(ious)

                IOU50_mask = iou >= 0.5
                iou50_success += IOU50_mask.sum().item()


                total_iou += iou.sum().item()
                count += len(iou)
            # --------------------
            # IoU计算（越大越好，重合程度高）
            # --------------------



        # --------------------
        # 最终统计
        # --------------------

        if pos_count > 0:
            avg_pos_conf = pos_conf_sum / pos_count
            pos_acc = pos_correct / pos_total
        else:
            avg_pos_conf = 0
            pos_acc = 0


        if neg_count > 0:
            avg_neg_conf = neg_conf_sum / neg_count
            neg_acc = neg_correct / neg_total
        else:
            avg_neg_conf = 0
            neg_acc = 0



        avg_iou = total_iou / count

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(
                mymoudle.state_dict(),
                "/kaggle/working/models/best_motor_detector.pth"
            )
            print(
                "保存最佳IoU模型"
            )
            
        avg_val_loss = val_loss_sum / len(ValDataloader)



        print("----------------验证结果----------------")

        # 1. 总Loss
        print(
            f"Val Loss(GIoU+Confidence): {avg_val_loss:.4f}"
        )


        # 2. BBox定位
        print(
            f"Mean IoU: {avg_iou:.4f}"
        )


        # 3. 置信度
        print(
            f"Positive Confidence(有目标): {avg_pos_conf:.4f}"
        )

        print(
            f"Negative Confidence(无目标): {avg_neg_conf:.4f}"
        )


        # 4. 分类准确率
        print(
            f"Positive Accuracy(有目标): {pos_acc:.4f}"
        )

        print(
            f"Negative Accuracy(无目标): {neg_acc:.4f}"
        )
        #5.Iou50的占比
        iou50_rate = iou50_success / count

        print(
    f"IoU@0.5成功数量: {iou50_success}/{count}, 成功率: {iou50_rate:.4f}"
)
        print("----------------------------------------")
