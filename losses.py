import torch
import torch.nn.functional as F

def relu_evidence(logits):
    return F.relu(logits)

def edl_mse_loss_with_prospect(
    logits,
    labels,
    epoch=None,
    num_classes=10,
    prior_strength=10,
    device='cpu',
    source_distribution=None,
    lambda_certainty=0.5
):
    evidence = relu_evidence(logits)                  # [batch_size, num_classes]
    alpha = evidence + 1                              # [batch_size, num_classes]
    S = torch.sum(alpha, dim=1, keepdim=True)        # [batch_size, 1]

    labels_one_hot = F.one_hot(labels.long(), num_classes=num_classes).float().to(device)  # [batch_size, num_classes]

    predicted = alpha / S                             # [batch_size, num_classes]

    # محاسبه mse به صورت sum روی کلاس‌ها برای هر نمونه و سپس میانگین روی batch
    mse_per_sample = torch.sum((labels_one_hot - predicted) ** 2, dim=1)  # [batch_size]
    mse = torch.mean(mse_per_sample)                                       # scalar

    uncertainty = num_classes / S                      # [batch_size, 1]
    certainty_penalty = torch.mean(uncertainty)       # scalar

    loss = mse + lambda_certainty * certainty_penalty

    return loss, mse.item(), certainty_penalty.item()
