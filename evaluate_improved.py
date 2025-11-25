# evaluate_baseline.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from week2_cnn_train_improved import ImprovedCNN  # import your CNN class

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load the saved model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()

# === Load test dataset ===
# Use same normalization as training for consistency
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dir = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test"
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# === Evaluate on test data ===
criterion = torch.nn.CrossEntropyLoss()
correct, total, test_loss = 0, 0, 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = test_loss / len(test_loader)
accuracy = correct / total

print("\n✅ Evaluation Complete!")
print(f"🖼️ Total Test Images: {total}")
print(f"🎯 Test Accuracy: {accuracy:.4f}")
print(f"📉 Average Test Loss: {avg_loss:.4f}")
