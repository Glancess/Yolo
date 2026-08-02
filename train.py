
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import YoloDataset
from model import MymoduleforYolo
from loss import DetectionLoss
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
MotorDataloader=DataLoader(train_dataset,batch_size=32,drop_last=True,shuffle=True)

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
    mymoudle.eval()
    val_loss_sum = 0


    total_iou = 0
    count = 0


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



            # --------------------
            # IoU计算
            # --------------------

            mask = target[:,4] == 1


            if mask.sum() > 0:

                pred_bbox = pred[mask,:4]

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


                total_iou += iou.sum().item()

                count += len(iou)



        avg_iou = total_iou / count

        avg_val_loss = val_loss_sum / len(ValDataloader)


        print(
            f"验证集平均Loss：{avg_val_loss:.4f}"
        )

        print(
            f"验证集平均IoU：{avg_iou:.4f}"
        )
