from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision


class YoloDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_name = sorted(
            path.name
            for path in Path(image_folder).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        # self.class_list=["no helmet","motor","number","with helmet"]

    def __getitem__(self, index):
        img_name = self.image_name[index]
        img_path = Path(self.image_folder) / img_name
        img = Image.open(img_path).convert("RGB")
        label_name = Path(img_name).stem + ".txt"
        label_path = Path(self.label_folder) / label_name

        targets = []

        with open(label_path, "r") as f:

            for line in f.readlines():

                process_i = line.strip().split()

                class_id = int(process_i[0])

                x = float(process_i[1])
                y = float(process_i[2])
                w = float(process_i[3])
                h = float(process_i[4])

                targets.append([class_id, x, y, w, h])

        if len(targets) == 0:

            target = torch.zeros((0, 5), dtype=torch.float32)
        else:
            target = torch.tensor(targets, dtype=torch.float32)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.image_name)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "datasets" / "helmet_yolo_train"
    train_dataset = YoloDataset(
        data_dir / "images",
        data_dir / "labels",
        torchvision.transforms.ToTensor(),
        None,
    )
    img, target = train_dataset[110]
    print(len(train_dataset))
    print(target)
