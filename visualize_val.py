# import torch
# import cv2
# import torchvision.transforms as transforms

# from PIL import Image

# from model import MymoduleforYolo



# def xywh_to_xyxy(box):

#     x,y,w,h = box

#     x1 = int((x-w/2)*512)
#     y1 = int((y-h/2)*512)

#     x2 = int((x+w/2)*512)
#     y2 = int((y+h/2)*512)


#     return x1,y1,x2,y2



# device="cuda"


# model=MymoduleforYolo()


# model.load_state_dict(
#     torch.load(
#         "/kaggle/working/models/best_motor_detector.pth"
#     )
# )


# model.to(device)

# model.eval()



# transform=transforms.Compose([
#     transforms.Resize((512,512)),
#     transforms.ToTensor()
# ])


# img_path="/kaggle/input/datasets/aleneger/yolo-conf/img1.jpg"


# img=Image.open(img_path).convert("RGB")


# tensor=transform(img)

# tensor=tensor.unsqueeze(0).to(device)



# with torch.no_grad():

#     pred=model(tensor)


# bbox=torch.sigmoid(
#     pred[0,:4]
# ).cpu()



# conf=torch.sigmoid(
#     pred[0,4]
# ).item()



# print(
#     "confidence:",
#     conf
# )



# # PIL转opencv

# img=cv2.imread(img_path)

# img=cv2.resize(
#     img,
#     (512,512)
# )



# if conf>0.5:


#     x1,y1,x2,y2 = xywh_to_xyxy(
#         bbox
#     )


#     cv2.rectangle(
#         img,
#         (x1,y1),
#         (x2,y2),
#         (0,0,255),
#         2
#     )


#     cv2.putText(
#         img,
#         f"{conf:.2f}",
#         (x1,y1-10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0,0,255),
#         2
#     )


# cv2.imwrite(
#     "result.png",
#     img
# )、

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

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]


    x1 = int((x - w / 2) * 512)
    y1 = int((y - h / 2) * 512)

    x2 = int((x + w / 2) * 512)
    y2 = int((y + h / 2) * 512)


    return x1, y1, x2, y2



# =========================
# 路径
# =========================

IMAGE_PATH = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/images"

LABEL_PATH = "/kaggle/input/datasets/aleneger/yolo-conf/helmet_yolo_val/labels"


MODEL_PATH = "/kaggle/working/models/best_motor_detector.pth"


SAVE_PATH = "/kaggle/working/visual_result"


os.makedirs(
    SAVE_PATH,
    exist_ok=True
)



# =========================
# device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# =========================
# transform
# =========================

transform = transforms.Compose([

    transforms.Resize((512,512)),

    transforms.ToTensor()

])



# =========================
# Dataset
# =========================

val_dataset = YoloDataset(

    IMAGE_PATH,

    LABEL_PATH,

    transform,

    None

)



# =========================
# DataLoader
# 随机取20张
# =========================

VisualLoader = DataLoader(

    val_dataset,

    batch_size=20,

    shuffle=True,

    drop_last=False

)



# =========================
# 加载模型
# =========================

model = MymoduleforYolo()


model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=device

    )

)


model.to(device)

model.eval()



# =========================
# 开始可视化
# =========================

with torch.no_grad():


    for imgs, targets in VisualLoader:


        imgs = imgs.to(device)


        preds = model(imgs)



        # batch里面20张

        for i in range(len(imgs)):


            img_tensor = imgs[i].cpu()


            target = targets[i]


            pred = preds[i].cpu()



            # -----------------
            # Tensor -> OpenCV
            # -----------------

            img_show = img_tensor.permute(
                1,2,0
            ).numpy()


            img_show = (
                img_show * 255
            ).astype("uint8")


            img_show = cv2.cvtColor(

                img_show,

                cv2.COLOR_RGB2BGR

            )



            # -----------------
            # GT框
            # -----------------

            target_conf = target[4]


            if target_conf == 1:


                gt_box = xywh_to_xyxy(

                    target[:4]

                )


                cv2.rectangle(

                    img_show,

                    (gt_box[0],gt_box[1]),

                    (gt_box[2],gt_box[3]),

                    (0,255,0),

                    2

                )


                cv2.putText(

                    img_show,

                    "GT",

                    (gt_box[0],gt_box[1]-5),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0,255,0),

                    2

                )



            # -----------------
            # 预测框
            # -----------------

            pred_bbox = torch.sigmoid(

                pred[:4]

            )


            conf = torch.sigmoid(

                pred[4]

            ).item()



            if conf > 0.5:


                pred_box = xywh_to_xyxy(

                    pred_bbox

                )


                cv2.rectangle(

                    img_show,

                    (pred_box[0],pred_box[1]),

                    (pred_box[2],pred_box[3]),

                    (0,0,255),

                    2

                )


                cv2.putText(

                    img_show,

                    f"Pred:{conf:.2f}",

                    (pred_box[0],pred_box[1]-20),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0,0,255),

                    2

                )



            # 保存

            save_file = os.path.join(

                SAVE_PATH,

                f"{i}.png"

            )


            cv2.imwrite(

                save_file,

                img_show

            )


            print(

                save_file,

                "confidence:",

                round(conf,3)

            )



        # 只跑第一个batch

        break



print("可视化完成")