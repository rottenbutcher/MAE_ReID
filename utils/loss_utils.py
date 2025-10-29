from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Compute the batch-hard triplet loss.

    Args:
        embeddings: Tensor of shape (B, D) containing feature embeddings.
        labels: Tensor of shape (B,) containing integer class labels.
        margin: Margin parameter for the triplet loss.

    Returns:
        A scalar tensor representing the batch-hard triplet loss. If valid
        positive or negative pairs do not exist within the batch the loss is
        returned as zero to avoid propagating NaNs.
    """
    if embeddings.ndim != 2:
        embeddings = embeddings.view(embeddings.size(0), -1)

    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.view(-1)
    if labels.size(0) != embeddings.size(0):
        raise ValueError("Labels and embeddings must have the same batch size.")

    device = embeddings.device
    labels = labels.to(device)

    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_neg = ~mask_pos

    # remove self-comparisons from positive mask
    mask_pos.fill_diagonal_(False)

    if not mask_pos.any() or not mask_neg.any():
        return embeddings.new_tensor(0.0)

    dist_ap = pairwise_dist.clone()
    dist_ap[~mask_pos] = float('-inf')
    hardest_pos = dist_ap.max(dim=1)[0]

    dist_an = pairwise_dist.clone()
    dist_an[~mask_neg] = float('inf')
    hardest_neg = dist_an.min(dim=1)[0]

    valid_pos = mask_pos.sum(dim=1) > 0
    valid_neg = mask_neg.sum(dim=1) > 0
    valid_mask = valid_pos & valid_neg

    if not valid_mask.any():
        return embeddings.new_tensor(0.0)

    hardest_pos = hardest_pos[valid_mask]
    hardest_neg = hardest_neg[valid_mask]

    loss = F.relu(hardest_pos - hardest_neg + margin)
    if loss.numel() == 0:
        return embeddings.new_tensor(0.0)
    return loss.mean()


def classification_triplet_loss(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    ce_criterion,
    *,
    features: Optional[torch.Tensor] = None,
    margin: float = 0.3,
    ce_weight: float = 1.0,
    triplet_weight: float = 1.0,
    normalize_triplet: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Combine classification loss with batch-hard triplet loss."""
    labels = labels.long()
    ce_loss = ce_criterion(log_probs, labels)
    total_loss = ce_weight * ce_loss
    triplet_loss = None

    if features is not None and triplet_weight > 0:
        embeddings = features
        if normalize_triplet:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        triplet_loss = batch_hard_triplet_loss(embeddings, labels, margin)
        if triplet_loss is not None:
            total_loss = total_loss + triplet_weight * triplet_loss

    return total_loss, ce_loss, triplet_loss
