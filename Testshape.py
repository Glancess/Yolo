import torch

from model import MymoduleforYolo
from loss import DetectionLoss
from Testdataloder import val_loader
import torch.optim as optim

# =====================
# device
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 模型
# =====================

mymodule = MymoduleforYolo()

mymodule.to(device)


# =====================
# loss
# =====================

criterion = DetectionLoss()


# =====================
# 测试
# =====================
optimizer = optim.Adam(mymodule.parameters(), lr=1e-4)  # 优化器

for epoch in range(2):

    mymodule.train()

    loss_sum = 0

    for imgs, targets in val_loader:

        imgs = imgs.to(device)

        targets = [t.to(device) for t in targets]

        pred = mymodule(imgs)

        pred = pred.reshape(pred.size(0), 100, 9)

        loss = criterion(pred, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

    print("epoch:", epoch, "loss:", loss_sum / len(val_loader))
