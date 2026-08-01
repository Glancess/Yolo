import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

from model import MymoduleforYolo


IMAGE_PATH="/kaggle/input/datasets/aleneger/yolo-conf/000009.png"


transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])


device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


img = Image.open(IMAGE_PATH).convert("RGB")

img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0).to(device)


model=MymoduleforYolo()

model.load_state_dict(
    torch.load(
        "/kaggle/working/models/best_motor_detector.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()



with torch.no_grad():

    pred=model(img_tensor)

    print("raw prediction:")
    print(pred)


    # -----------------------
    # bbox
    # -----------------------

    bbox = torch.sigmoid(pred[:,:4])


    # -----------------------
    # confidence
    # -----------------------

    conf = torch.sigmoid(pred[:,4])


    print("bbox:",bbox)
    print("confidence:",conf)


    threshold = 0.5


    if conf[0] < threshold:

        print("没有检测到摩托")
        

    else:

        print("检测到摩托")


        x = bbox[0][0].item()*512
        y = bbox[0][1].item()*512
        w = bbox[0][2].item()*512
        h = bbox[0][3].item()*512


        x1=int(x-w/2)
        y1=int(y-h/2)

        x2=int(x+w/2)
        y2=int(y+h/2)


        img=cv2.imread(IMAGE_PATH)

        img=cv2.resize(img,(512,512))


        cv2.rectangle(
            img,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.imwrite(
            "/kaggle/working/result.png",
            img
        )

        print("保存结果")