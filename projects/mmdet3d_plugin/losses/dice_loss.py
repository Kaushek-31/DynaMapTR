import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
import torch.nn.functional as F

@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, 
                 eps=1e-6, 
                 loss_weight=1.0, 
                 class_weight=None,     # ← NEW
                 sigmoid=False, 
                 use_sigmoid=False, 
                 **kwargs):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        
        # If no weights provided → use equal weights
        self.class_weight = class_weight

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices
        """
        pred_soft = torch.softmax(pred, dim=1)
        num_classes = pred_soft.shape[1]

        # If config did not provide weights → ones
        if self.class_weight is None:
            weights = torch.ones(num_classes, device=pred.device)
        else:
            weights = torch.tensor(self.class_weight, device=pred.device)

        total_loss = 0.0
        weight_sum = weights.sum()

        for c in range(num_classes):
            pred_c = pred_soft[:, c, :, :]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() + self.eps

            dice = 2 * intersection / union
            dice_loss = (1 - dice)

            # ---- APPLY CLASS WEIGHT ----
            total_loss += weights[c] * dice_loss

        # Normalize by total weight
        total_loss = total_loss / weight_sum

        return self.loss_weight * total_loss

@LOSSES.register_module()
class BalancedCELoss(nn.Module):
    """Weighted Cross Entropy with optional focal behavior."""
    def __init__(self, class_weight=None, loss_weight=1.0, gamma=0.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma

        if class_weight is not None:
            self.register_buffer(
                "class_weight",
                torch.tensor(class_weight, dtype=torch.float32)
            )
        else:
            self.class_weight = None

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        B, C, H, W = pred.shape
        log_probs = torch.log_softmax(pred, dim=1)               # [B, C, H, W]
        log_probs = log_probs.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
        target_flat = target.reshape(-1)                         # [B*H*W]

        weight = self.class_weight if self.class_weight is not None else None
        ce = nn.NLLLoss(weight=weight)(log_probs, target_flat)

        if self.gamma > 0:
            pt = torch.exp(-ce)
            ce = ((1 - pt) ** self.gamma) * ce

        return self.loss_weight * ce


@LOSSES.register_module()
class GeneralizedDiceLoss(nn.Module):
    """Class-size balanced Dice loss."""
    def __init__(self, loss_weight=1.0, smooth=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W]
        B, C, H, W = pred.size()
        pred_soft = torch.softmax(pred, dim=1)              # [B, C, H, W]

        # One-hot GT
        target_onehot = torch.zeros_like(pred_soft)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0) # [B, C, H, W]

        # Class weights inversely proportional to squared volume
        w = 1.0 / (target_onehot.sum(dim=(0, 2, 3))**2 + self.smooth)  # [C]

        intersection = (pred_soft * target_onehot).sum(dim=(0, 2, 3))  # [C]
        union = pred_soft.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))  # [C]

        dice = (2.0 * intersection * w) / (union * w + self.smooth)    # [C]
        loss = 1.0 - dice.mean()
        return self.loss_weight * loss


@LOSSES.register_module()
class SegmentationHybridLoss(nn.Module):
    """
    CE + Generalized Dice, returns a single scalar tensor.
    """
    def __init__(self,
                 ce_class_weight=[0.1, 5.0, 5.0],  # bg, veh, ped
                 ce_weight=1.0,
                 dice_weight=1.0,
                 gamma=0.0,        # set >0 if you want focal behavior
                 loss_weight=1.0, **kwargs): # global weight if needed
        super().__init__()
        self.ce = BalancedCELoss(
            class_weight=ce_class_weight,
            loss_weight=ce_weight,
            gamma=gamma
        )
        self.gdl = GeneralizedDiceLoss(loss_weight=dice_weight)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.gdl(pred, target)
        total = loss_ce + loss_dice
        return self.loss_weight * total   # <- SINGLE tensor


@LOSSES.register_module()
class FocalDiceLoss(nn.Module):
    def __init__(self,
                 class_weights=[0.2, 1.0, 4.0],   # bg, vehicle, pedestrian
                 gamma=2.0,
                 dice_weight=1.0,
                 focal_weight=1.0, **kwargs):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def dice_loss(self, pred_soft, target):
        total = 0
        num_classes = pred_soft.shape[1]

        for c in range(num_classes):
            pred_c = pred_soft[:, c]
            tgt_c  = (target == c).float()

            inter = (pred_c * tgt_c).sum(dim=[1,2])
            union = pred_c.sum(dim=[1,2]) + tgt_c.sum(dim=[1,2]) + 1e-6

            dice = 1 - (2 * inter / union)
            total += dice.mean()

        return total / num_classes

    def focal_loss(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal

    def forward(self, logits, target):
        pred_soft = torch.softmax(logits, dim=1)

        loss_dice = self.dice_loss(pred_soft, target)
        loss_focal = self.focal_loss(logits, target)

        return self.dice_weight * loss_dice + self.focal_weight * loss_focal


@LOSSES.register_module()
class FocalDiceLossV2(nn.Module):
    """
    Fixed Focal + Dice loss with proper per-pixel computation.
    """
    def __init__(self,
                 class_weights=[0.1, 2.0, 4.0],  # bg, vehicle, pedestrian
                 gamma=2.0,
                 alpha=0.25,  # Focal alpha
                 dice_weight=1.0,
                 focal_weight=1.0,
                 smooth=1e-6,
                 **kwargs):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        
        # Register as buffer so it moves to correct device
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def focal_loss(self, logits, target):
        """Proper per-pixel focal loss."""
        # logits: (B, C, H, W), target: (B, H, W)
        B, C, H, W = logits.shape
        
        # Compute per-pixel CE loss (no reduction)
        ce_loss = F.cross_entropy(
            logits, target, 
            weight=self.class_weights,
            reduction='none'
        )  # (B, H, W)
        
        # Compute pt (probability of true class)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Gather probability of the target class
        target_expanded = target.unsqueeze(1)  # (B, 1, H, W)
        pt = probs.gather(1, target_expanded).squeeze(1)  # (B, H, W)
        
        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha > 0:
            # More weight to foreground (non-background)
            alpha_weight = torch.where(target > 0, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        
        focal_loss = (focal_weight * ce_loss).mean()
        
        return focal_loss

    def dice_loss(self, logits, target):
        """Per-class dice loss with class weighting."""
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        B, C, H, W = probs.shape
        
        # One-hot encode target
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)  # (B, C, H, W)
        
        # Compute per-class dice
        dims = (0, 2, 3)  # Sum over batch, height, width
        
        intersection = (probs * target_onehot).sum(dim=dims)  # (C,)
        union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)  # (C,)
        
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)  # (C,)
        
        # Apply class weights to dice
        if self.class_weights is not None:
            # Weight the dice loss (not the dice score)
            dice_loss_per_class = (1 - dice_per_class) * self.class_weights
            dice_loss = dice_loss_per_class.sum() / self.class_weights.sum()
        else:
            dice_loss = 1 - dice_per_class.mean()
        
        return dice_loss

    def forward(self, logits, target):
        loss_focal = self.focal_loss(logits, target)
        loss_dice = self.dice_loss(logits, target)
        
        total = self.focal_weight * loss_focal + self.dice_weight * loss_dice
        return total