import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tenseal as ts
from week2_cnn_train_improved import ImprovedCNN  # use your improved CNN
import numpy as np

# === Device setup ===
device = torch.device("cpu")
print("Using device:", device)

# === Load trained CNN model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# === Image preprocessing (same as training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load one sample test image ===
img_path = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"
image = Image.open(img_path).convert("RGB")
x = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
print("🖼️ Image preprocessed:", x.shape)

# === Plain prediction ===
with torch.no_grad():
    plain_output = model(x)
    plain_pred = torch.argmax(plain_output, dim=1).item()
print(f"🧠 Plain prediction: {plain_pred} (0=Normal, 1=Pneumonia)")

# === Extract features after convolution layers ===
with torch.no_grad():
    features = model.conv(x)  # shape: (1, 64, 28, 28)
    features_flat = features.view(-1).tolist()
print("📦 Feature vector extracted for encryption:", len(features_flat), "values")

# === Create TenSEAL encryption context ===
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# === Encrypt feature vector ===
enc_input = ts.ckks_vector(context, features_flat)
print("🔐 Feature vector encrypted successfully!")

# === Simulate encrypted linear layer ===
fc1_weight = model.fc[0].weight.data.numpy()
fc1_bias = model.fc[0].bias.data.numpy()

# Encrypt bias term too
enc_bias = ts.ckks_vector(context, [fc1_bias[0]])

# Perform encrypted dot product + bias addition
enc_output = enc_input.dot(fc1_weight[0]) + enc_bias
print("🔒 Encrypted computation (dot product + bias) done successfully.")

# Decrypt to verify correctness
decrypted_output = enc_output.decrypt()[0]
print(f"🔓 Decrypted result (sample neuron output): {decrypted_output:.6f}")


print("\n✅ Encrypted inference simulation complete!")
