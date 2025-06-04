import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from matplotlib import pyplot as plt
import json
from helpers import get_device
from data import dataloaders
from train import train_model
from losses import edl_mse_loss, edl_mse_loss_with_prospect
from prospect_certainty import refine_logits_with_prospect_certainty


def main():
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="To train the network.")
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument("--examples", action="store_true", help="To example data.")

    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs (default=30).")
    parser.add_argument("--dropout", action="store_true", help="Use dropout or not.")
    parser.add_argument("--uncertainty", action="store_true", help="Use uncertainty or not.")

    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", action="store_true",
        help="Use Expected Mean Square Error loss (for uncertainty).")

    args = parser.parse_args()

    device = get_device()

    if args.examples:
        # نمایش چند نمونه تصویر از دیتاست اعتبارسنجی
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title(f"Ground Truth: {example_targets[i]}")
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")
        print("Saved examples to ./images/examples.jpg")

    elif args.train:
        num_classes = 10
        num_epochs = args.epochs

        # تعریف مدل
        model = ResNet18(num_classes=num_classes, dropout=args.dropout)
        model = model.to(device)

        # انتخاب criterion
        if args.uncertainty:
            if args.mse:
                criterion = edl_mse_loss_with_prospect
            else:
                parser.error("--uncertainty requires --mse.")
        else:
            criterion = nn.CrossEntropyLoss()

        # بهینه‌ساز و scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # محاسبه توزیع برچسب‌ها روی دیتاست آموزش (برای Prospect Certainty)
        label_counts = torch.zeros(num_classes)
        for _, labels in dataloaders["train"]:
            for label in labels:
                label_counts[label] += 1
        source_distribution = (label_counts / label_counts.sum()).tolist()
        with open("./results/source_distribution.json", "w") as f:
            json.dump(source_distribution, f)
        print("Saved source_distribution to ./results/source_distribution.json")

        # آموزش مدل
        model, metrics = train_model(
            model=model,
            dataloaders=dataloaders,
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=args.uncertainty,
            source_distribution=source_distribution,
        )

        # ذخیره بهترین مدل
        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if args.uncertainty and args.mse:
            torch.save(state, "./results/model_uncertainty_mse.pt")
            print("Saved model to ./results/model_uncertainty_mse.pt")
        else:
            torch.save(state, "./results/model.pt")
            print("Saved model to ./results/model.pt")

    elif args.test:
        num_classes = 10
        model = ResNet18(num_classes=num_classes, dropout=args.dropout)
        model = model.to(device)

        if args.uncertainty and args.mse:
            checkpoint = torch.load("./results/model_uncertainty_mse.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model_uncertainty_mse.pt for testing")
        else:
            checkpoint = torch.load("./results/model.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model.pt for testing")

        model.eval()

        with open("./results/source_distribution.json", "r") as f:
            source_distribution = json.load(f)

        # ارزیابی مدل با prospect certainty
        refined_outputs, certainty_scores = refine_logits_with_prospect_certainty(
            model, dataloaders["val"], source_distribution, device=device
        )

        # دقت مدل با refined outputs
        preds = torch.argmax(refined_outputs, dim=1).cpu()
        targets = []
        for _, labels in dataloaders["val"]:
            targets.extend(labels.cpu().numpy())
        targets = torch.tensor(targets)

        accuracy = (preds == targets).float().mean().item()
        print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
