import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import f1_score, accuracy_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1-score: {f1:.4f}")
    return acc, f1


def main():
    device = get_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # بارگذاری وزن‌های مدل ذخیره شده
    checkpoint_path = "checkpoints/model_cifar10.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}")

    # فقط تست مدل بدون آموزش
    test_acc, test_f1 = test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
