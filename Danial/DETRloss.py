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
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids].log() # ------------------------------------------------------------ calculates the "how wrong the model's classification is" = measures confidence

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # ---------------------------------------------------- calcs the "how wring the model's box location is compared to the ground truth" - Compute negative log-likelihood classification cost (higher for wrong predictions) = uses simple l1 distance, absolute difference between coordinates


        # Compute the GIoU cost between boxes
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        ) # --------------------------------------------------------------------------------------------------- same as before, measures how wrong the box location is, but considering the spaces between the boxes - generalised intersction over union = gives sxore of -1 to +1

        # Final cost matrix
        C = (self.matcher_cost_bbox * cost_bbox +
             self.matcher_cost_class * cost_class +
             self.matcher_cost_giou * cost_giou)
        C = C.view(batch_size, num_queries, -1).cpu() # ------------------------------------------------------- combines all the three above costs into one single cost matrix

        sizes = [len(v["boxes"]) for v in targets]
        indices = []

        for i, c in enumerate(C.split(sizes, -1)):
            # Hungarian algorithm on the detached CPU tensor
            pred_indices, target_indices = linear_sum_assignment(c[i].detach().cpu().numpy()) # ---------------- looks at the entire cost matrix and finds the single best way to pair the predicted objects = weighted sum of three individual costs
            indices.append((torch.as_tensor(pred_indices, dtype=torch.int64),
                          torch.as_tensor(target_indices, dtype=torch.int64)))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

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

    def generalized_box_iou(self, boxes1, boxes2): # -------------------------------------------------------------------------------------------------------- explain the intersection over union formula -
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
        iou = inter / union

        # Compute the area of the smallest enclosing box
        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        areai = whi[:, :, 0] * whi[:, :, 1]

        return iou - (areai - union) / areai

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
                - boxes: Tensor of dim [num_objects, 4] containing the boxes in cx,cy,w,h format

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
# criterion = DETRLoss(model2.class_embed.out_features - 1) # really just 11 output dims
def loss_fn(model):
    criterion = DETRLoss(
        num_classes=model.class_embed.out_features - 1, # exclude the background class
        matcher_cost_class=1,
        matcher_cost_bbox=5,
        matcher_cost_giou=2,
        loss_ce=1,
        loss_bbox=5,
        loss_giou=2,
        eos_coef=0.1
    )
    return criterion