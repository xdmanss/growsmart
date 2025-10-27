from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time

# === Paths (relative to this script) ===
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
MODEL_SAVE_PATH = BASE_DIR / "growsmart_mnv2.pth"

# === Hyperparameters ===
BATCH_SIZE = 32          # if you get CUDA OOM, try 16
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2          # set to 0 if you see Windows DataLoader issues

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check folders
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("❌ 'train'/'val' folders not found. Run dataset_split.py first.")
        return

    # Transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & loaders
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tfms)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    start_all = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * running_correct / total
        train_time = time.time() - start

        # ---- Validate ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%  (epoch time {train_time:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_ds.classes,
                "arch": "mobilenet_v2"
            }, MODEL_SAVE_PATH)
            print(f"💾 Saved best model → {MODEL_SAVE_PATH.name}  (Val Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_all
    print(f"🎉 Training complete in {total_time/60:.1f} min. Best Val Acc: {best_val_acc:.2f}%.")

if __name__ == "__main__":
    main()