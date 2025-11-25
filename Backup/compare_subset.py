import torch
from torchvision import transforms
from PIL import Image
import tenseal as ts
import numpy as np
import os, random, time
from week2_cnn_train_improved import ImprovedCNN

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load trained CNN model ===
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# === Select random image ===
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

# === Extract full feature vector ===
with torch.no_grad():
    features = model.conv(input_tensor)
    features = features.view(features.size(0), -1).cpu().numpy()[0]
total_features = len(features)
print(f"🧩 Total feature vector length: {total_features}")

# === Load FC1 weights and biases ===
fc1_weight = model.fc[0].weight.detach().cpu().numpy()
fc1_bias = model.fc[0].bias.detach().cpu().numpy()

# === Helper: Encrypted inference function (multi-ciphertext capable) ===
def encrypted_fc1_inference(features_subset, fc1_weight, fc1_bias, poly_mod_degree=16384):
    coeff_mod_bit_sizes = [40, 21, 21, 40]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    context.generate_galois_keys()
    context.global_scale = 2**21

    max_slots = poly_mod_degree // 2
    chunks = [features_subset[i:i+max_slots] for i in range(0, len(features_subset), max_slots)]

    start = time.time()
    enc_chunks = [ts.ckks_vector(context, c.tolist()) for c in chunks]
    encryption_time = time.time() - start

    enc_outputs = []
    for i in range(fc1_weight.shape[0]):  # each neuron
        total_val = 0.0
        for j, chunk in enumerate(chunks):
            w_chunk = fc1_weight[i, j*max_slots : j*max_slots + len(chunk)]
            if len(w_chunk) == 0:
                continue
            enc_dot = enc_chunks[j].dot(w_chunk.tolist())
            total_val += enc_dot.decrypt()[0]
        enc_outputs.append(total_val + float(fc1_bias[i]))

    inference_time = time.time() - start
    return np.array(enc_outputs), encryption_time, inference_time

# === Benchmark different subset sizes ===
subset_sizes = [1024, 2048, 4096, 8192, total_features]
results = []

for subset_size in subset_sizes:
    print(f"\n🔹 Running encrypted inference for subset {subset_size} features...")
    features_subset = features[:subset_size]
    fc1_weight_sub = fc1_weight[:, :subset_size]

    # Plain FC1 inference
    t0 = time.time()
    plain_output = np.dot(fc1_weight_sub, features_subset) + fc1_bias
    plain_time = time.time() - t0

    # Encrypted FC1 inference
    enc_output, enc_time, inf_time = encrypted_fc1_inference(features_subset, fc1_weight_sub, fc1_bias)

    mse = np.mean((enc_output - plain_output) ** 2)
    corr = np.corrcoef(enc_output, plain_output)[0, 1]
    results.append((subset_size, enc_time, inf_time, mse, corr, plain_time))

# === Print results table ===
print("\n📊 Benchmark Results")
print("Subset | Encryption(s) | Total Encrypted(s) | MSE | Corr | Plain(s)")
print("-"*75)
for r in results:
    print(f"{r[0]:>6} | {r[1]:>13.2f} | {r[2]:>19.2f} | {r[3]:>8.3f} | {r[4]:>6.4f} | {r[5]:>8.3f}")
