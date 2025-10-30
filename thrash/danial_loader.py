import torchvision.transforms as transforms
import torch
import os
from torch.utils.data import DataLoader


ds = "Military objects in military environments/Dataset"
# ds = f"military drones images"

models = f"/mnt/d/23b6034/FYP/models/DETR-4"
train_dataset = f"/mnt/d/23b6034/FYP/datasets/{ds}/train"
test_dataset = f"/mnt/d/23b6034/FYP/datasets/{ds}/test"
val_dataset = f"/mnt/d/23b6034/FYP/datasets/{ds}/valid"

class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, classes, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt') # Assuming label files have the same name as images with a .txt extension

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, center_x, center_y, width, height = map(float, line.strip().split())
                    # Convert YOLO format (center_x, center_y, width, height) to [x_min, y_min, x_max, y_max]
                    # Assuming coordinates are normalized (0 to 1)
                    x_min = center_x - width / 2
                    y_min = center_y - height / 2
                    x_max = center_x + width / 2
                    y_max = center_y + height / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id)) # Assuming class_id is an integer

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return image, target
    
classes = {
    0 : "Tank",
    1 : "drone",
    2 : "people",
    3 : "soldier"
}
#tank drone people soldier

# Re-define the classes dictionary


# Define image and label directory paths
train_image_dir = train_dataset + "/images"
train_label_dir = train_dataset + "/labels"
val_image_dir = val_dataset + "/images"
val_label_dir = val_dataset + "/labels"
test_image_dir = test_dataset + "/images"
test_label_dir = test_dataset + "/labels"

# Define a suitable transform
# For DETR, a common transform resizes the image and normalizes it
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    # Add more transforms if needed, e.g., resizing to a fixed size expected by DETR
    # transforms.Resize((800, 800)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the DroneDataset for each set
train_drone_dataset = DroneDataset(train_image_dir, train_label_dir, classes, transform=transform)
val_drone_dataset = DroneDataset(val_image_dir, val_label_dir, classes, transform=transform)
test_drone_dataset = DroneDataset(test_image_dir, test_label_dir, classes, transform=transform)


# Instantiate DataLoader objects for each dataset
batch_size = 1  # You can adjust this based on your GPU memory
# Added num_workers for parallel data loading
def load_train():
    train_dataloader = DataLoader(train_drone_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)
def load_val():
    val_dataloader = DataLoader(val_drone_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)
def load_test():
    test_dataloader = DataLoader(test_drone_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)

# print("DataLoaders for train, validation, and test datasets instantiated with num_workers.")