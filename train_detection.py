import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# -----------------------
# Custom Dataset Class
# -----------------------
class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        """
        Args:
            root (string): Root directory of the dataset split (e.g., 'reduced/train').
                           It should contain subfolders 'images' and 'annotations'.
            transforms (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.anns = sorted(os.listdir(os.path.join(root, "annotations")))
        # Assumes image filenames and annotation filenames (except extension) match

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.anns[idx])
        img = Image.open(img_path).convert("RGB")

        # Parse annotation XML to get bounding boxes
        tree = ET.parse(ann_path)
        root_xml = tree.getroot()
        boxes = []
        for obj in root_xml.findall("object"):
            # All objects will be labeled as "lego" (label = 1)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Set label 1 for all objects
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])
        if boxes.numel() > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

# -----------------------
# Data Transforms
# -----------------------
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Data augmentation: random horizontal flip with probability 0.5
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# -----------------------
# Collate Function (Top-Level)
# -----------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# -----------------------
# Main Training Function
# -----------------------
def main():
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)
    
    # Create training and validation datasets
    dataset_train = LegoDataset(root="reduced/train", transforms=get_transform(train=True))
    dataset_val = LegoDataset(root="reduced/val", transforms=get_transform(train=False))
    
    # Create data loaders using the top-level collate function
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
        
    print("Loading pre-trained model...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    num_classes = 2  # background and lego
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    # Construct optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    print("Starting training...")
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs} started...")
            model.train()
            epoch_loss = 0
            batch_count = 0
            for batch_idx, (images, targets) in enumerate(data_loader_train):
                batch_count += 1
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                epoch_loss += loss_value

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Debug: print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss_value:.4f}")

            lr_scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    
    # Save the trained model
    torch.save(model.state_dict(), "fasterrcnn_lego.pth")
    print("Training complete. Model saved as 'fasterrcnn_lego.pth'.")

if __name__ == "__main__":
    main()