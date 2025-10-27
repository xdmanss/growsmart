from pathlib import Path
import shutil
import random

# === Paths (relative to this script) ===
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "full_dataset"   # <- your class folders live directly inside this
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# === Settings ===
SPLIT_RATIO = 0.8          # 80% train / 20% val
RANDOM_SEED = 42           # for reproducibility
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def create_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    random.seed(RANDOM_SEED)

    if not DATASET_DIR.exists():
        print(f"❌ full_dataset folder not found at: {DATASET_DIR}")
        print("Make sure your structure is: growsmart-ml/full_dataset/<class folders>")
        return

    create_dir(TRAIN_DIR)
    create_dir(VAL_DIR)

    class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        print("❌ No class folders found inside full_dataset. Each class should be a folder with images.")
        return

    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        if not images:
            print(f"⚠️  Skipping empty class folder: {class_dir.name}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Make class subfolders in train/ and val/
        train_cls = TRAIN_DIR / class_dir.name
        val_cls = VAL_DIR / class_dir.name
        create_dir(train_cls)
        create_dir(val_cls)

        for src in train_imgs:
            dst = train_cls / src.name
            shutil.copy2(src, dst)

        for src in val_imgs:
            dst = val_cls / src.name
            shutil.copy2(src, dst)

        print(f"✅ {class_dir.name}: {len(train_imgs)} train, {len(val_imgs)} val")

    print("\n🎉 Split complete! 'train/' and 'val/' are ready.")

if __name__ == "__main__":
    main()