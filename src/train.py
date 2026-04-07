"""
Training Script — Face Mask Detection
Usage:
    python src/train.py --data_dir data/ --epochs 20 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from model import MaskDetector
from dataset import get_dataloaders


# ── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Mask Detector")
    parser.add_argument("--data_dir",   default="data/",      help="Dataset root dir")
    parser.add_argument("--epochs",     type=int, default=20,  help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,  help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir",   default="models/",     help="Model save directory")
    parser.add_argument("--plot_dir",   default="plots/",      help="Plots output directory")
    return parser.parse_args()


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metrics(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Face Mask Detection — Training Metrics", fontsize=14, fontweight="bold")

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved training curves → {save_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved confusion matrix → {save_dir}/confusion_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    # Model
    model = MaskDetector(num_classes=2, pretrained=True).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  Training for {args.epochs} epochs | Batch: {args.batch_size}")
    print(f"{'='*55}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:>2}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "mask_detector.pth"))
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # Final report
    print(f"\n{'='*55}")
    print("Classification Report (Validation Set):")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # Plots
    plot_metrics(history, args.plot_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, args.plot_dir)

    print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Model saved → {args.save_dir}/mask_detector.pth")


if __name__ == "__main__":
    main()
