import torch
import torch.nn.functional as F

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot_embedding(labels, num_classes, smoothing=0.0):
    """
    تبدیل لیبل‌های عددی به one-hot با امکان label smoothing
    """
    assert labels.dim() == 1
    y = torch.zeros(labels.size(0), num_classes, device=labels.device)
    y.scatter_(1, labels.unsqueeze(1), 1)
    if smoothing > 0:
        y = y * (1 - smoothing) + smoothing / num_classes
    return y


def relu_evidence(y):
    """
    تبدیل خروجی مدل به evidence با استفاده از ReLU
    """
    return F.relu(y)
