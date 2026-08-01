
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import YoloDataset
from model import MymoduleforYolo
from loss import DetectionLoss

import torch.optim as optim

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

    with torch.no_grad():

        for img, target in ValDataloader:

            img = img.to(device)
            target = target.to(device)

            pred = mymoudle(img)

            loss = criterion(pred, target)

            val_loss_sum += loss.item()

    avg_val_loss = val_loss_sum / len(ValDataloader)

    print(f"验证集平均Loss：{avg_val_loss:.4f}")

    if avg_val_loss < best_loss:

        best_loss = avg_val_loss

        torch.save(
            mymoudle.state_dict(),
            "/kaggle/working/models/best_motor_detector.pth",
        )

        print("✅ 保存最佳模型")
