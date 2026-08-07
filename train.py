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
from val import validate


def collate_fn(batch):

    imgs = []
    targets = []

    for img, target in batch:

        imgs.append(img)
        targets.append(target)

    imgs = torch.stack(imgs)
    # 因为图片可以堆叠，只关心 target

    return imgs, targets


transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]
)
train_dataset = YoloDataset(
    "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_train/images",
    "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_train/labels",
    transform,
    None,
)

# 对于 一张图片有4个物体，也就是返回的size=4*5.另一张是 2*5.无法stack，而batch想要一批大小相同的，对应label也是一一对应的。


MotorDataloader = DataLoader(
    train_dataset, batch_size=16, drop_last=False, shuffle=True, collate_fn=collate_fn
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymoudle = MymoduleforYolo()  # 模型
criterion = DetectionLoss()  # 损失
mymoudle.to(device)

for param in mymoudle.img_contract.parameters():
    param.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, mymoudle.parameters()), lr=1e-4
)


best_iou50 = 0  # 衡量

for epoch in range(30):
    print(f"--------开始第 {epoch} 轮训练----------")
    loss_sum = 0
    mymoudle.train()
    for img, target in MotorDataloader:

        img = img.to(device)

        target = [t.to(device) for t in target]

        pred = mymoudle(img)

        pred = pred.reshape(pred.size(0), 100, 9)

        loss = criterion(pred, target)
        loss_sum += loss.item()
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print("train loss:", loss_sum / len(MotorDataloader))

    # ===================
    # val
    # ===================

    result = validate(mymoudle, device)
    print(result)
    # train只负责训练模型，改变参数，验证分开

    # ===================
    # save best
    # ===================
    if result["iou50"] > best_iou50:

        best_iou50 = result["iou50"]

        torch.save(mymoudle.state_dict(), "/kaggle/working/best_yolo.pth")

        print("保存最佳模型")
