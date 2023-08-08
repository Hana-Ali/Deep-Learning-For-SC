import torch
import torch.nn as nn

# Define the contrastive loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
# Define the max-margin contrastive loss
class MaxMarginContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaxMarginContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative samples
        labels_matrix = labels.view(-1, 1)
        positive_mask = labels_matrix == labels_matrix.t()
        negative_mask = labels_matrix != labels_matrix.t()

        # Remove the diagonal elements (self-pairs)
        positive_mask.fill_diagonal_(0)

        pos_dists = torch.zeros(batch_size).to(embeddings.device)
        neg_dists = torch.zeros(batch_size).to(embeddings.device)

        for i in range(batch_size):
            pos_distances_i = distances[i][positive_mask[i]]
            neg_distances_i = distances[i][negative_mask[i]]

            pos_dists[i] = pos_distances_i.min() if pos_distances_i.numel() > 0 else float('inf')
            neg_dists[i] = neg_distances_i.max() if neg_distances_i.numel() > 0 else float('-inf')

        # Compute max-margin loss
        loss = torch.clamp(pos_dists - neg_dists + self.margin, min=0.0).mean()

        return loss

import torch
import torch.nn as nn

class ContrastiveLossWithPosNegPairs(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossWithPosNegPairs, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Compute masks for positive and negative pairs
        labels_matrix = labels.view(-1, 1)
        positive_mask = labels_matrix == labels_matrix.t()
        negative_mask = labels_matrix != labels_matrix.t()

        # Remove the diagonal elements (self-pairs)
        positive_mask.fill_diagonal_(0)

        # Initialize loss
        loss = 0

        # Iterate through the batch and find the first example that has both positive and negative pairs
        for i in range(batch_size):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]

            # Check if there is at least one positive and one negative sample
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Select the closest positive and the furthest negative sample
                pos_dist = distances[i, pos_indices].min()
                neg_dist = distances[i, neg_indices].max()

                # Compute the contrastive loss for this example
                loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                break

        return loss

