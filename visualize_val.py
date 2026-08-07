import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import os

from dataset import YoloDataset
from model import MymoduleforYolo

# =========================
# xywh -> xyxy
# =========================


def xywh_to_xyxy(box):

    x, y, w, h = box

    x1 = int((x - w / 2) * 512)
    y1 = int((y - h / 2) * 512)

    x2 = int((x + w / 2) * 512)
    y2 = int((y + h / 2) * 512)

    return x1, y1, x2, y2


# =========================
# 类别
# =========================

classes = ["no helmet", "motor", "number", "with helmet"]


IMAGE_PATH = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/images"

LABEL_PATH = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/labels"


MODEL_PATH = "/kaggle/working/best_yolo.pth"


SAVE_PATH = "/kaggle/working/visual_result"


os.makedirs(SAVE_PATH, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])


dataset = YoloDataset(IMAGE_PATH, LABEL_PATH, transform, None)


def collate_fn(batch):

    imgs = []
    targets = []

    for img, target in batch:

        imgs.append(img)
        targets.append(target)

    return torch.stack(imgs), targets


loader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)


# =========================
# model
# =========================

model = MymoduleforYolo()


model.load_state_dict(torch.load(MODEL_PATH, map_location=device))


model.to(device)

model.eval()


# =========================
# visualize
# =========================

with torch.no_grad():

    for imgs, targets in loader:

        imgs = imgs.to(device)

        preds = model(imgs)

        # B,900 -> B,100,9

        preds = preds.reshape(preds.size(0), 100, 9)

        for i in range(len(imgs)):

            img = imgs[i].cpu()

            img = img.permute(1, 2, 0).numpy()

            img = (img * 255).astype("uint8")

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # ======================
            # GT
            # ======================

            target = targets[i]

            for gt in target:

                cls = int(gt[0])

                box = xywh_to_xyxy(gt[1:5])

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                cv2.putText(
                    img,
                    "GT:" + classes[cls],
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # ======================
            # Prediction 100 boxes
            # ======================

            pred = preds[i].cpu()
            pred_conf = torch.sigmoid(pred[:, 4])

            print("confidence:")
            print(pred_conf)

            print("max confidence:", pred_conf.max().item())

            print("top5 confidence:", torch.topk(pred_conf, 5))

            for box_pred in pred:

                # bbox

                bbox = torch.sigmoid(box_pred[:4])

                conf = torch.sigmoid(box_pred[4]).item()

                # 类别

                cls_prob = torch.softmax(box_pred[5:], dim=0)

                cls_id = torch.argmax(cls_prob).item()

                if conf < 0.2:
                    continue

                box = xywh_to_xyxy(bbox)

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                cv2.putText(
                    img,
                    f"{classes[cls_id]} {conf:.2f}",
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            save = os.path.join(SAVE_PATH, f"{i}.png")

            cv2.imwrite(save, img)

            print(save)

        break


print("可视化完成")
