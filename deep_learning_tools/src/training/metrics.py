import torch

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the accuracy metric.

    Args:
        outputs (torch.Tensor): Model outputs (N, C) where C is number of classes
        targets (torch.Tensor): Target labels (N,) containing class indices

    Returns:
        float: Accuracy percentage (0-100)
    """
    _, preds = torch.max(outputs, dim=1)
    return (torch.sum(preds == targets).item() / targets.size(0)) * 100


def precision(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate precision for both binary and multi-class classifications.
    
    Args:
        outputs: Model outputs (N, C) where C is 1 for binary or num_classes for multi-class
        targets: Target labels (N,) containing class indices
    Returns:
        float: Precision score as percentage (0-100)
    """
    # handle binary case (outputs shape is [N, 1])
    if outputs.shape[1] == 1:
        preds = (outputs >= 0.5).squeeze().float()
        true_positives = ((preds == 1) & (targets == 1)).sum().float()
        predicted_positives = (preds == 1).sum().float()
        precision_val = (true_positives / predicted_positives) if predicted_positives > 0 else torch.tensor(0.0)
        return precision_val.item() * 100
    
    # handle multi-class case
    _, preds = torch.max(outputs, dim=1)
    true_positives = torch.zeros(outputs.shape[1], device=outputs.device)
    predicted_positives = torch.zeros(outputs.shape[1], device=outputs.device)
    
    for c in range(outputs.shape[1]):
        true_positives[c] = ((preds == c) & (targets == c)).sum().float()
        predicted_positives[c] = (preds == c).sum().float()
    
    # macro-averaging
    class_precisions = torch.where(predicted_positives > 0, 
                                 true_positives / predicted_positives, 
                                 torch.zeros_like(predicted_positives))
    return class_precisions.mean().item() * 100


def recall(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate recall for both binary and multi-class classifications.
    
    Args:
        outputs: Model outputs (N, C) where C is 1 for binary or num_classes for multi-class
        targets: Target labels (N,) containing class indices
    Returns:
        float: Recall score as percentage (0-100)
    """
    # handle binary case (outputs shape is [N, 1])
    if outputs.shape[1] == 1:
        preds = (outputs >= 0.5).squeeze().float()
        true_positives = ((preds == 1) & (targets == 1)).sum().float()
        actual_positives = (targets == 1).sum().float()
        recall_val = (true_positives / actual_positives) if actual_positives > 0 else torch.tensor(0.0)
        return recall_val.item() * 100
    
    # handle multi-class case
    _, preds = torch.max(outputs, dim=1)
    true_positives = torch.zeros(outputs.shape[1], device=outputs.device)
    actual_positives = torch.zeros(outputs.shape[1], device=outputs.device)
    
    for c in range(outputs.shape[1]):
        true_positives[c] = ((preds == c) & (targets == c)).sum().float()
        actual_positives[c] = (targets == c).sum().float()
    
    # macro-averaging
    class_recalls = torch.where(actual_positives > 0,
                              true_positives / actual_positives,
                              torch.zeros_like(actual_positives))
    return class_recalls.mean().item() * 100


def f1_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate F1 score for both binary and multi-class classifications.
    
    Args:
        outputs: Model outputs (N, C) where C is 1 for binary or num_classes for multi-class
        targets: Target labels (N,) containing class indices
    Returns:
        float: F1 score as percentage (0-100)
    """
    p = precision(outputs, targets)
    r = recall(outputs, targets)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0