from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2

# ===== Paths =====
MODEL_PATH = Path("growsmart_mnv2.pth")  # trained model file
IMAGE_PATH = Path("tomato_leaf.jpg")      # replace with your image filename

# ===== Load model checkpoint =====
print("🔸 Loading model checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
class_names = checkpoint["class_names"]

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===== Image transforms =====
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===== Load and preprocess image =====
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

print(f"📸 Loading image: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert("RGB")
x = transform(img).unsqueeze(0)  # Add batch dimension

# ===== Make prediction =====
with torch.no_grad():
    outputs = model(x)
    _, pred = torch.max(outputs, 1)
    predicted_class = class_names[pred.item()]

print(f"✅ Prediction: {predicted_class}")
