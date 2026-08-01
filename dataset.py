
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision 


class YoloDataset(Dataset):
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder=image_folder
        self.label_folder=label_folder
        self.transform=transform
        self.label_transform=label_transform
        self.image_name=sorted(
            path.name
            for path in Path(image_folder).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        #self.class_list=["no helmet","motor","number","with helmet"]
    def __getitem__(self,index):
        img_name=self.image_name[index]
        img_path = Path(self.image_folder) / img_name
        img=Image.open(img_path).convert("RGB")
        label_name=Path(img_name).stem + ".txt"
        label_path = Path(self.label_folder) / label_name

        with open(label_path,"r",encoding="utf-8") as f:
            label_content=f.read()


# ============================
# 负样本：空label
# ============================
            if label_content.strip() == "":
                
                target = torch.tensor([
                    0,  # x
                    0,  # y
                    0,  # w
                    0,  # h
                    0,  # confidence
                ], dtype=torch.float32)


            else:

                Label_contents = label_content.strip().split("\n")

                target = None


                for line in Label_contents:

                    process_i = line.strip().split()

                    class_id = int(process_i[0])


                    # 只保留motor
                    if class_id != 0:
                        continue


                    x = float(process_i[1])
                    y = float(process_i[2])
                    w = float(process_i[3])
                    h = float(process_i[4])


                    target = torch.tensor(
                        [
                            x,
                            y,
                            w,
                            h,
                            1,      
                        ],
                        dtype=torch.float32
                    )

                    break


                if target is None:
                    raise ValueError(
                        f"No motor annotation found in {label_path}"
                    )


            img=self.transform(img)

            return img,target
    
    def __len__(self):
        return len(self.image_name)

if __name__=='__main__':
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "datasets" / "helmet_yolo_train"
    train_dataset=YoloDataset(data_dir / "images", data_dir / "labels", torchvision.transforms.ToTensor(),None)
    img,target=train_dataset[110]
    print(len(train_dataset))
    print(target)
