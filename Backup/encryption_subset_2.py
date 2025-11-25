import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tenseal as ts
import numpy as np
import os, random
from week2_cnn_train_improved import ImprovedCNN  # import your CNN model

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load trained CNN model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# === Randomly select one image from Pneumonia test folder ===
folder = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test\PNEUMONIA"
img_path = os.path.join(folder, random.choice(os.listdir(folder)))
print("🖼️ Selected image:", img_path)

# === Preprocess image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
print("🧪 Image preprocessed successfully.")

# === Plain model full inference ===
with torch.no_grad():
    plain_output = model(input_tensor)
plain_output = plain_output.squeeze().cpu().numpy()
print(f"🧠 Plain model final output: {plain_output}")

# === Extract features before FC layer ===
with torch.no_grad():
    features = model.conv(input_tensor)
    features = features.view(features.size(0), -1).cpu().numpy()[0]
print(f"🧩 Feature vector extracted: {len(features)} values")

# === Use only a subset of features for encryption (for CKKS limits) ===
subset_size = 4092   # can safely encrypt this many values
features_subset = features[:subset_size]
print(f"🔹 Using subset of {subset_size} features for encrypted inference")

# === Get weights/bias from first FC layer (truncate to match subset) ===
fc1_weight = model.fc[0].weight.detach().cpu().numpy()[:, :subset_size]
fc1_bias = model.fc[0].bias.detach().cpu().numpy()

# === Initialize CKKS encryption context ===
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 40]
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
context.generate_galois_keys()
context.global_scale = 2**21

# === Encrypt subset of feature vector ===
enc_input = ts.ckks_vector(context, features_subset.tolist())
print("🔒 Subset of feature vector encrypted successfully.")

# === Compute encrypted outputs for FC1 layer ===
enc_outputs = []
for i in range(fc1_weight.shape[0]):
    w_i = fc1_weight[i].tolist()
    b_i = float(fc1_bias[i])
    # encrypted × plaintext dot
    enc_result = enc_input.dot(w_i)
    # decrypt and add bias
    decrypted_val = enc_result.decrypt()[0] + b_i
    enc_outputs.append(decrypted_val)

enc_outputs = np.array(enc_outputs)
print(f"🔓 Decrypted encrypted FC1 layer outputs shape: {enc_outputs.shape}")

# === Compute plain outputs for same subset ===
with torch.no_grad():
    plain_fc1_output = np.dot(fc1_weight, features_subset) + fc1_bias

# === Compare plain vs encrypted ===
mse = np.mean((enc_outputs - plain_fc1_output) ** 2)
corr = np.corrcoef(enc_outputs, plain_fc1_output)[0, 1]

print("\n✅ Encrypted subset inference complete!")
print(f"Plain FC1 output shape: {plain_fc1_output.shape}")
print(f"Decrypted FC1 output shape: {enc_outputs.shape}")
print(f"MSE: {mse:.8e}")
print(f"Correlation: {corr:.6f}")
