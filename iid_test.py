import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from prospect_certainty import refine_logits_with_prospect_certainty
from helpers import get_device
from resnet import ResNet18
from data import dataloaders
import json
from torch.utils.data import TensorDataset, DataLoader

def test_model_with_certainty():
    device = get_device()
    model = ResNet18(num_classes=10, dropout=False).to(device)

    # Load trained model
    checkpoint = torch.load("./results/model_uncertainty_mse.pt", map_location=device)
# اگر checkpoint خودش state_dict است:
    model.load_state_dict(checkpoint)
    model.eval()

    with open("./results/source_distribution.json", "r") as f:
        source_distribution = json.load(f)

    val_loader = dataloaders["val"]

    print("\U0001F50D Evaluating model with Prospect Certainty...")

    # Run inference and compute refined logits and certainty
    refined_outputs, certainty_scores = refine_logits_with_prospect_certainty(
        model, val_loader, source_distribution, device
    )

    # Predicted labels from refined outputs
    preds = torch.argmax(refined_outputs, dim=1).cpu()
    labels = torch.cat([label for _, label in val_loader]).cpu()

    f1 = f1_score(labels, preds, average='macro')
    print(f"✅ CIFAR-10 Test F1-score (Refined with Prospect Certainty): {f1:.4f}")

    # Display a few samples with certainty
    num_samples = 8
    sample_inputs, sample_labels = next(iter(val_loader))
    sample_inputs, sample_labels = sample_inputs.to(device), sample_labels.to(device)

    sample_dataset = TensorDataset(sample_inputs, sample_labels)
    sample_loader = DataLoader(sample_dataset, batch_size=sample_inputs.size(0))

    with torch.no_grad():
        refined, certainties = refine_logits_with_prospect_certainty(model, sample_loader, source_distribution, device)

    sample_preds = torch.argmax(refined, dim=1)
    sample_certainties = certainties[:num_samples].cpu().numpy()

    data = {
        "True Label": sample_labels[:num_samples].cpu().numpy(),
        "Predicted Label": sample_preds[:num_samples].cpu().numpy(),
        "Certainty Score": [f"{c:.3f}" for c in sample_certainties]
    }
    df = pd.DataFrame(data)
    print("\n📊 Sample Prediction Table with Certainty:")
    print(df)

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    ax.table(cellText=df.values,
             colLabels=df.columns,
             cellLoc='center',
             loc='center',
             colColours=["#cccccc"] * df.shape[1])
    plt.title("Prediction Table with Prospect Certainty", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig("certainty_cifar10_predictions.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_model_with_certainty()
