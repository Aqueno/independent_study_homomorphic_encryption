import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tenseal as ts
import numpy as np
import os, random, time
import matplotlib.pyplot as plt             # for the scatter plot
from week2_cnn_train_improved import ImprovedCNN  # import your CNN model


# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load trained CNN model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.\n")

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
    # flatten to 1-D feature vector (length should be 50176)
    features = features.view(features.size(0), -1).cpu().numpy()[0]
print(f"🧩 Feature vector extracted: {len(features)} values\n")

# === Get FC1 weights and bias ===
fc1_weight = model.fc[0].weight.detach().cpu().numpy()  # shape (128, 50176)
fc1_bias = model.fc[0].bias.detach().cpu().numpy()

# === Initialize CKKS encryption context ===
poly_mod_degree = 16384
coeff_mod_bit_sizes = [40, 21, 21, 40]
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
context.generate_galois_keys()
context.global_scale = 2**21

# === Multi-ciphertext encryption ===
chunk_size = 8192
num_chunks = int(np.ceil(len(features) / chunk_size))
enc_chunks = []

start_enc = time.time()
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(features))
    chunk = features[start:end]
    enc_chunks.append(ts.ckks_vector(context, chunk.tolist()))
enc_time_encrypt = time.time() - start_enc

print(f"🔐 Encrypted {num_chunks} ciphertext chunks (chunk size ≤ {chunk_size})")
print(f"Total encrypted features: {len(features)}")
print(f"⏱️ Encryption time: {enc_time_encrypt:.2f} sec\n")

# === Encrypted inference (multi-ciphertext dot product) ===
enc_outputs = []
start_infer = time.time()
for i in range(fc1_weight.shape[0]):
    b_i = float(fc1_bias[i])
    total_val = 0.0
    for j in range(num_chunks):
        w_chunk = fc1_weight[i][j * chunk_size:(j + 1) * chunk_size]
        if len(w_chunk) == 0:
            continue
        enc_dot = enc_chunks[j].dot(w_chunk.tolist())
        # Decrypt partial result and sum (prototype design)
        total_val += enc_dot.decrypt()[0]
    enc_outputs.append(total_val + b_i)
enc_outputs = np.array(enc_outputs)
enc_time_infer = time.time() - start_infer

print(f"🔓 Decrypted encrypted FC1 outputs shape: {enc_outputs.shape}")
print(f"⏱️ Encrypted inference time: {enc_time_infer:.2f} sec\n")

# === Compute plain outputs for same FC1 layer ===
with torch.no_grad():
    feat_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    plain_fc1_output = model.fc[0](feat_tensor).squeeze().cpu().numpy()

# === Scatter plot: Plain vs Encrypted FC1 activations ===
plt.figure(figsize=(5, 5))
plt.scatter(plain_fc1_output, enc_outputs, s=20)

min_val = min(plain_fc1_output.min(), enc_outputs.min())
max_val = max(plain_fc1_output.max(), enc_outputs.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

plt.xlabel("Plain FC1 output")
plt.ylabel("Encrypted (decrypted) FC1 output")
plt.title("Plain vs Encrypted FC1 Activations (Single Image)")
plt.tight_layout()
plt.savefig("fc1_scatter.png", dpi=300)
plt.show()

# === Compare plain vs encrypted ===
mse = np.mean((enc_outputs - plain_fc1_output) ** 2)
corr = np.corrcoef(enc_outputs, plain_fc1_output)[0, 1]

total_runtime = enc_time_encrypt + enc_time_infer

print("✅ Multi-ciphertext encrypted inference complete!\n")
print(f"Plain FC1 output shape: {plain_fc1_output.shape}")
print(f"Decrypted FC1 output shape: {enc_outputs.shape}")
print(f"MSE: {mse:.8f}")
print(f"Correlation: {corr:.6f}")
print(f"⚙️ Total runtime (encrypt + infer): {total_runtime:.2f} sec")
