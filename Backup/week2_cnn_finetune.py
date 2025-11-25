import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from week2_cnn_train_improved import ImprovedCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load dataset again ===
data_dir = r"C:\Users\niras\Desktop\independent_study\data\chest_xray"
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

train_dataset = datasets.ImageFolder(f"{data_dir}\\train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{data_dir}\\val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Load trained model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
print("✅ Loaded trained model for fine-tuning.")

# === Loss, optimizer, and scheduler ===
weights = torch.tensor([1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# === Fine-tuning for 5 epochs ===
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()
    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Train acc: {train_acc:.3f} | Loss: {total_loss/len(train_loader):.3f}")

print("✅ Fine-tuning complete.")
torch.save(model.state_dict(), "week2_cnn_finetuned.pth")
