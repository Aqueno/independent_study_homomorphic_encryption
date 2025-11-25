# week2_cnn_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Define Enhanced CNN model ===
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === Train only when run directly ===
if __name__ == "__main__":
    # === Data transforms with augmentation + normalization ===
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # === Load dataset ===
    data_dir = r"C:\Users\niras\Desktop\independent_study\data\chest_xray"
    train_dataset = datasets.ImageFolder(f"{data_dir}\\train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}\\val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # === Initialize model, loss, optimizer ===
    model = ImprovedCNN().to(device)

    # Handle class imbalance
    weights = torch.tensor([1.0, 1.5]).to(device)  # tweak if needed
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)

    # === Training loop ===
    num_epochs = 15
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)

        # === Validation ===
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f} | "
              f"Loss: {total_loss/len(train_loader):.3f} | "
              f"Time: {time.time()-start:.1f}s")

    print("✅ Training complete.")
    torch.save(model.state_dict(), "week2_cnn_improved.pth")
