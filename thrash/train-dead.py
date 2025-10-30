import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import sys
from datetime import datetime
log_file_path = "log.txt"

def setup_environment():
    """Setup environment and check for CUDA availability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return device

class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, classes, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.transform = transform
        
        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
            
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(self.image_filenames) == 0:
            raise ValueError(f"No image files found in {image_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        try:
            img_name = self.image_filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            boxes = []
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:  # Skip empty lines
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id, center_x, center_y, width, height = map(float, parts[:5])
                                # Convert YOLO format to [x_min, y_min, x_max, y_max]
                                x_min = center_x - width / 2
                                y_min = center_y - height / 2
                                x_max = center_x + width / 2
                                y_max = center_y + height / 2
                                boxes.append([x_min, y_min, x_max, y_max])
                                labels.append(int(class_id))

            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

            return image, target
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Return a dummy item to prevent DataLoader from crashing
            dummy_image = torch.zeros((3, 224, 224))
            dummy_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
            return dummy_image, dummy_target

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class DETRLoss(nn.Module):
    """
    DETR Loss Function with Hungarian Matching

    This loss computes the optimal bipartite matching between predicted and ground truth objects,
    and then computes classification and bounding box regression losses.
    """

    def __init__(self, num_classes, matcher_cost_class=1, matcher_cost_bbox=5,
                 matcher_cost_giou=2, loss_ce=2, loss_bbox=2.5, loss_giou=2,
                 eos_coef=0.1):
        """
        Parameters:
        - num_classes: number of object categories
        - matcher_cost_class: relative weight of classification error in matching cost
        - matcher_cost_bbox: relative weight of L1 error of bounding box coordinates in matching
        - matcher_cost_giou: relative weight of giou loss of bounding box in matching
        - loss_ce: relative weight of classification loss
        - loss_bbox: relative weight of L1 bounding box loss
        - loss_giou: relative weight of giou bounding box loss
        - eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher_cost_class = matcher_cost_class
        self.matcher_cost_bbox = matcher_cost_bbox
        self.matcher_cost_giou = matcher_cost_giou
        self.loss_ce = loss_ce
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.eos_coef = eos_coef

        # Build weight vector for classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # Background class
        self.register_buffer('empty_weight', empty_weight)

    def hungarian_matching(self, outputs, targets):
        """
        Performs the matching using Hungarian algorithm

        Args:
            outputs: Dict with 'pred_logits' [batch_size, num_queries, num_classes + 1]
                    and 'pred_boxes' [batch_size, num_queries, 4]
            targets: List of dicts, each containing 'labels' and 'boxes' for each image

        Returns:
            List of tuples (pred_idx, target_idx) for each image in the batch
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids].log()

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the GIoU cost between boxes
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (self.matcher_cost_bbox * cost_bbox +
             self.matcher_cost_class * cost_class +
             self.matcher_cost_giou * cost_giou)
        C = C.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = outputs["pred_logits"].device

        # for i, c in enumerate(C.split(sizes, -1)):
        #     # Hungarian algorithm on the detached CPU tensor
        #     pred_indices, target_indices = linear_sum_assignment(c[i].detach().cpu().numpy())
        #     indices.append((torch.as_tensor(pred_indices, dtype=torch.int64),
        #                   torch.as_tensor(target_indices, dtype=torch.int64)))

        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        #         for i, j in indices]
        for i, c in enumerate(C.split(sizes, -1)):
            # Hungarian algorithm on the detached CPU tensor
            c_matrix = c[i].detach().cpu().numpy()
            pred_indices, target_indices = linear_sum_assignment(c_matrix)
            # Ensure indices are returned to the original device!
            indices.append((torch.as_tensor(pred_indices, dtype=torch.int64, device=device),
                          torch.as_tensor(target_indices, dtype=torch.int64, device=device)))
    
        return indices # This is now a list of tuples on the correct device

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (Cross Entropy)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # Ensure target_classes is on the same device as src_logits
        target_classes = target_classes.to(src_logits.device)

        # Ensure empty_weight is on the same device as src_logits
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / len(target_boxes) if len(target_boxes) > 0 else torch.tensor(0.0, device=src_boxes.device)

        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(src_boxes),
            self.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / len(target_boxes) if len(target_boxes) > 0 else torch.tensor(0.0, device=src_boxes.device)

        return loss_bbox, loss_giou

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def box_cxcywh_to_xyxy(self, x):
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        """
        # Ensure boxes are valid
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

        # Compute intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter

        # Compute IoU
        # iou = inter / union
        iou = inter / (union + 1e-8)

        # Compute the area of the smallest enclosing box
        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        # areai = whi[:, :, 0] * whi[:, :, 1]

        # return iou - (areai - union) / areai

        areai = whi[:, :, 0] * whi[:, :, 1]
        gious = iou - (areai - union) / (areai + 1e-8) # Another epsilon here
        return gious

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
            outputs: dict of tensors with keys:
                - pred_logits: Tensor of dim [batch_size, num_queries, num_classes + 1]
                - pred_boxes: Tensor of dim [batch_size, num_queries, 4] in cxcywh format
            targets: list of dicts, such that len(targets) == batch_size.
                Each dict should contain:
                - labels: Tensor of dim [num_objects] containing the class labels
                - boxes: Tensor of dim [num_objects, 4] containing the boxes in cxcywh format

        Returns:
            dict: A dictionary containing the losses
        """
        # Retrieve the matching between the outputs of the model and the targets
        indices = self.hungarian_matching(outputs, targets)

        # Compute all the losses
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)

        # Combine losses
        losses = {
            'loss_ce': loss_ce * self.loss_ce,
            'loss_bbox': loss_bbox * self.loss_bbox,
            'loss_giou': loss_giou * self.loss_giou,
        }

        # Total loss
        losses['loss_total'] = sum(losses.values())

        return losses

def custom_collate_fn(batch):
    """Custom collate function to handle variable sized targets"""
    images, targets = tuple(zip(*batch))
    return list(images), list(targets)

def load_model_and_datasets():
    """Load the DETR model and create datasets"""
    print("Loading DETR model...")
    try:
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None
    
    # Dataset paths
    train_dataset = "/workspace/military-drones-images-1/train"
    test_dataset = "/workspace/military-drones-images-1/test"
    val_dataset = "/workspace/military-drones-images-1/valid"
    
    # Re-define the classes dictionary
    classes = {
        0: "-", 1: "BBM", 2: "BMP", 3: "BTR", 4: "CAY",
        5: "MTLB", 6: "Shishiga", 7: "buhanka", 8: "soldier",
        9: "tank", 10: "truck"
    }

    # Define image and label directory paths
    train_image_dir = os.path.join(train_dataset, "images")
    train_label_dir = os.path.join(train_dataset, "labels")
    val_image_dir = os.path.join(val_dataset, "images")
    val_label_dir = os.path.join(val_dataset, "labels")
    test_image_dir = os.path.join(test_dataset, "images")
    test_label_dir = os.path.join(test_dataset, "labels")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    try:
        # Instantiate the DroneDataset for each set
        train_drone_dataset = DroneDataset(train_image_dir, train_label_dir, classes, transform=transform)
        val_drone_dataset = DroneDataset(val_image_dir, val_label_dir, classes, transform=transform)
        test_drone_dataset = DroneDataset(test_image_dir, test_label_dir, classes, transform=transform)
        
        print(f"Train dataset size: {len(train_drone_dataset)}")
        print(f"Validation dataset size: {len(val_drone_dataset)}")
        print(f"Test dataset size: {len(test_drone_dataset)}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None, None, None

    # Instantiate DataLoader objects
    batch_size = 4  # Reduced batch size for stability
    
    train_dataloader = DataLoader(
        train_drone_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=0,
        pin_memory=False
    )
    val_dataloader = DataLoader(
        val_drone_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=0,
        pin_memory=False
    )
    test_dataloader = DataLoader(
        test_drone_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=0,
        pin_memory=False
    )

    return model, train_dataloader, val_dataloader, test_dataloader

def train_model():
    """Main training function"""
    device = setup_environment()
    
    # Load model and data
    model, train_dataloader, val_dataloader, test_dataloader = load_model_and_datasets()
    if model is None:
        print("Failed to load model or datasets. Exiting.")
        return
    
    model.to(device)
    
    # Initialize loss function
    # criterion = DETRLoss(model.class_embed.out_features - 1)
    # In your criterion initialization, use these values:
    criterion = DETRLoss(
        num_classes=model.class_embed.out_features - 1, # This is correct
        matcher_cost_class=1,   # Keep this
        matcher_cost_bbox=5,    # Keep this
        matcher_cost_giou=2,    # Keep this - this is the paper baseline for matcher
        loss_ce=1,              # CHANGE TO 1 (from 2)
        loss_bbox=5,            # CHANGE TO 5 (from 2.5) - This forces precise box regression
        loss_giou=2,            # CHANGE TO 2 (from 2) - Keep it, it's good.
        eos_coef=0.1            # This is the standard value
    )
    
    # Training parameters
    num_epochs = 300
    checkpoint_dir = "/workspace/military-drones-images-1/models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    optimizer = None

    # Check for existing checkpoints
    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            print("Found existing checkpoints. Resuming training.")
            latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getctime)
            print(f"Loading checkpoint from: {latest_checkpoint}")
            
            checkpoint_data = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data)
            
            # Extract epoch number
            start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print("No existing checkpoints found. Starting training from scratch.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch.")

    # Initialize optimizer
    optimizer = optim.AdamW([
        {'params': model.class_embed.parameters()},
        {'params': model.bbox_embed.parameters()}
    ], lr=5e-6)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        try:
            model.train()
            running_loss = 0.0
            num_batches = 0
            
            print(f"\nStarting epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                try:
                    # Skip empty batches
                    if not images or all(len(t['labels']) == 0 for t in targets):
                        continue

                    # Move data to device
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # Forward pass
                    outputs = model(images)
                    
                    # Compute loss
                    loss_dict = criterion(outputs, targets)
                    loss = loss_dict['loss_total']
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss detected at batch {batch_idx}. Skipping.")
                        continue

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1
                    
                except Exception as batch_error:
                    print(f"Error in batch {batch_idx}: {batch_error}")
                    continue

            if num_batches > 0:
                avg_loss = running_loss / num_batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], No valid batches processed")

            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n=== Training Completed at {timestamp}  with loss {avg_loss:.4f}  for epoch {epoch+1} ===\n")
                    log_file.flush()
            except Exception as log_error:
                print(f"Warning: Could not write completion message to log file: {log_error}")

            # Save checkpoint
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
                
                # Delete previous previous model (keep only current and previous)
                if epoch >= 1:  # Only delete if we have at least 2 models
                    previous_previous_model = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                    if os.path.exists(previous_previous_model):
                        try:
                            os.remove(previous_previous_model)
                            print(f"Deleted old checkpoint: {previous_previous_model}")
                        except Exception as delete_error:
                            print(f"Warning: Could not delete old checkpoint {previous_previous_model}: {delete_error}")
                            
            except Exception as save_error:
                print(f"Error saving model: {save_error}")

        except Exception as epoch_error:
            print(f"Error during epoch {epoch+1}: {epoch_error}")
            print("Attempting to continue with next epoch...")
            continue

    print("Training completed!")

def main():
    """Main function"""
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()