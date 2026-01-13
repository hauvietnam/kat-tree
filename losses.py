"""
Advanced Loss Functions for Imbalanced Classification
- LDAM Loss (Label-Distribution-Aware Margin Loss)
- Asymmetric Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss
    Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (NeurIPS 2019)
    
    This loss is designed for long-tailed/imbalanced datasets by introducing 
    class-dependent margins that are inversely proportional to the class frequency.
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        Args:
            cls_num_list: list or array, number of samples for each class
            max_m: float, maximum margin value (default: 0.5)
            weight: torch.Tensor, optional class weights for additional weighting
            s: float, scale parameter (default: 30)
        """
        super(LDAMLoss, self).__init__()
        # Calculate margin for each class based on frequency
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        """
        Args:
            x: torch.Tensor, logits with shape (batch_size, num_classes)
            target: torch.Tensor, ground truth labels with shape (batch_size,)
        Returns:
            loss: torch.Tensor, scalar loss value
        """
        # Move m_list to same device as x
        if self.m_list.device != x.device:
            self.m_list = self.m_list.to(x.device)
        
        # Create index mask for target classes
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        # Calculate margin for each sample based on its class
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        
        # Apply margin to target class logits
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        
        # Scale and compute cross entropy
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Paper: Asymmetric Loss For Multi-Label Classification (ICCV 2021)
    
    Can be adapted for single-label classification with class imbalance by
    applying different focusing parameters to positive and negative samples.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 disable_torch_grad_focal_loss=True):
        """
        Args:
            gamma_neg: float, focusing parameter for negative samples (default: 4)
            gamma_pos: float, focusing parameter for positive samples (default: 1)
            clip: float, probability clipping parameter (default: 0.05)
            eps: float, numerical stability constant (default: 1e-8)
            disable_torch_grad_focal_loss: bool, whether to disable gradient for focal term
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor, logits with shape (batch_size, num_classes)
            y: torch.Tensor, labels
               - For single-label: shape (batch_size,) with class indices
               - For multi-label: shape (batch_size, num_classes) with 0/1 values
        Returns:
            loss: torch.Tensor, scalar loss value
        """
        # Convert single-label to multi-label format if needed
        if y.dim() == 1:
            num_classes = x.shape[1]
            y_one_hot = torch.zeros_like(x)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            y = y_one_hot

        # Calculate probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.shape[0]


class AsymmetricLossSingleLabel(nn.Module):
    """
    Asymmetric Loss adapted specifically for single-label classification
    with long-tail distribution.
    
    This version is optimized for single-label scenarios and provides
    better control over the focusing mechanism.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, eps=1e-8, reduction='mean'):
        """
        Args:
            gamma_neg: float, focusing parameter for negative classes (default: 4)
            gamma_pos: float, focusing parameter for positive class (default: 0)
            eps: float, numerical stability constant (default: 1e-8)
            reduction: str, 'mean' or 'sum' (default: 'mean')
        """
        super(AsymmetricLossSingleLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape (batch_size, num_classes)
            targets: torch.Tensor, shape (batch_size,) with class indices
        Returns:
            loss: torch.Tensor, scalar loss value
        """
        num_classes = logits.shape[1]
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Separate positive and negative probabilities
        pos_probs = probs * targets_one_hot
        neg_probs = probs * (1 - targets_one_hot)
        
        # Calculate asymmetric focal weights
        pos_weight = torch.pow(1 - pos_probs, self.gamma_pos)
        neg_weight = torch.pow(neg_probs, self.gamma_neg)
        
        # Calculate cross entropy with focal weighting
        log_probs = F.log_softmax(logits, dim=1)
        
        # Positive samples
        pos_loss = -targets_one_hot * log_probs * pos_weight
        # Negative samples  
        neg_loss = -(1 - targets_one_hot) * log_probs * neg_weight
        
        loss = pos_loss.sum(dim=1) + neg_loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined LDAM and Asymmetric Loss
    
    Combines the benefits of both losses:
    - LDAM: Class-dependent margins based on frequency
    - Asymmetric: Differential treatment of positive/negative samples
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, 
                 gamma_neg=4, gamma_pos=1, 
                 lambda_ldam=0.5, lambda_asym=0.5):
        """
        Args:
            cls_num_list: list, number of samples per class
            max_m: float, maximum margin for LDAM (default: 0.5)
            s: float, scale parameter for LDAM (default: 30)
            gamma_neg: float, negative focusing for asymmetric loss (default: 4)
            gamma_pos: float, positive focusing for asymmetric loss (default: 1)
            lambda_ldam: float, weight for LDAM loss (default: 0.5)
            lambda_asym: float, weight for asymmetric loss (default: 0.5)
        """
        super(CombinedLoss, self).__init__()
        self.ldam_loss = LDAMLoss(cls_num_list, max_m=max_m, s=s)
        self.asym_loss = AsymmetricLossSingleLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        self.lambda_ldam = lambda_ldam
        self.lambda_asym = lambda_asym
    
    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape (batch_size, num_classes)
            targets: torch.Tensor, shape (batch_size,)
        Returns:
            loss: torch.Tensor, combined loss value
        """
        loss_ldam = self.ldam_loss(logits, targets)
        loss_asym = self.asym_loss(logits, targets)
        loss = self.lambda_ldam * loss_ldam + self.lambda_asym * loss_asym
        return loss, loss_ldam, loss_asym


# Utility functions
def get_cls_num_list(dataset):
    """
    Extract class distribution from dataset
    
    Args:
        dataset: PyTorch dataset with targets or labels attribute
    Returns:
        list: number of samples for each class
    """
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")
    
    num_classes = len(np.unique(targets))
    cls_num_list = [np.sum(targets == i) for i in range(num_classes)]
    return cls_num_list


def get_cls_num_list_from_loader(train_loader):
    """
    Extract class distribution from DataLoader
    
    Args:
        train_loader: PyTorch DataLoader
    Returns:
        list: number of samples for each class
    """
    if hasattr(train_loader.dataset, 'targets'):
        return get_cls_num_list(train_loader.dataset)
    
    # Alternative: iterate through loader (slower)
    cls_counts = {}
    for _, labels in train_loader:
        for label in labels:
            label = label.item()
            cls_counts[label] = cls_counts.get(label, 0) + 1
    
    num_classes = len(cls_counts)
    cls_num_list = [cls_counts.get(i, 0) for i in range(num_classes)]
    return cls_num_list