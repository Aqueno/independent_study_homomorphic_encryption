import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import tenseal as ts
import warnings
warnings.filterwarnings("ignore")


from week2_cnn_train_improved import ImprovedCNN

# --- setup ---
device = torch.device("cpu")
print("Using device:", device)

# model
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()

# data (same normalization as training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dir = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test"
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# HE context (CKKS)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# get weights for the first neuron in fc1
fc1_weight = model.fc[0].weight.data.numpy()    # shape: (128, 50176)
fc1_bias   = model.fc[0].bias.data.numpy()      # shape: (128,)
w0 = fc1_weight[0]
b0 = fc1_bias[0]

plain_vals = []
enc_vals   = []

# run on first N images
N = 10
with torch.no_grad():
    for i, (img, _) in enumerate(test_loader):
        if i >= N: break
        # conv features
        feats = model.conv(img.to(device))                 # (1, 64, 28, 28)
        feats_flat = feats.view(-1).cpu().numpy()          # (50176,)
        # plain dot + bias (reference)
        plain_out = float(np.dot(feats_flat, w0) + b0)
        plain_vals.append(plain_out)

        # encrypted dot + bias
        enc_in   = ts.ckks_vector(context, feats_flat.tolist())
        enc_bias = ts.ckks_vector(context, [b0])
        enc_out  = enc_in.dot(w0) + enc_bias               # encrypted scalar
        dec_out  = enc_out.decrypt()[0]
        enc_vals.append(dec_out)

# compare
plain_vals = np.array(plain_vals)
enc_vals   = np.array(enc_vals)
mse  = np.mean((plain_vals - enc_vals)**2)
corr = np.corrcoef(plain_vals, enc_vals)[0,1]

print("\n✅ Comparison over", N, "images")
print("   Plain mean:", plain_vals.mean())
print("   Encrypted(decrypted) mean:", enc_vals.mean())
print("   MSE:", mse)
print("   Correlation:", corr)
