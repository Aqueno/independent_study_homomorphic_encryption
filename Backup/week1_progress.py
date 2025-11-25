import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import tenseal as ts

print("✅ Environment setup complete")

# --- Load dataset ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    r"C:\Users\niras\Desktop\independent_study\data\chest_xray\train",
    transform=transform
)

print(f"Dataset loaded: {len(dataset)} images, classes: {dataset.classes}")

# Save a random sample image
img, label = random.choice(dataset)
plt.imshow(img.permute(1, 2, 0))
plt.title(dataset.classes[label])
plt.axis("off")
plt.savefig("sample_image.png")
print("Sample image saved as sample_image.png")

# --- Homomorphic Encryption demo ---
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 20, 40]
)

# ✅ set the global scale
context.global_scale = 2**40

# generate keys
context.generate_galois_keys()

# now encrypt
enc_a = ts.ckks_vector(context, [3.5])
enc_b = ts.ckks_vector(context, [2.5])
enc_sum = enc_a + enc_b
result = enc_sum.decrypt()

print("Decrypted sum:", result)

