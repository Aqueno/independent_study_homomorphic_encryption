import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Adjust to your dataset root
DATA_ROOT = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test"

normal_dir = os.path.join(DATA_ROOT, "NORMAL")
pneumonia_dir = os.path.join(DATA_ROOT, "PNEUMONIA")

normal_img_path = os.path.join(normal_dir, random.choice(os.listdir(normal_dir)))
pneumonia_img_path = os.path.join(pneumonia_dir, random.choice(os.listdir(pneumonia_dir)))

print("Normal sample:", normal_img_path)
print("Pneumonia sample:", pneumonia_img_path)

# Save individual images (for separate use if needed)
Image.open(normal_img_path).save("normal_sample.png")
Image.open(pneumonia_img_path).save("pneumonia_sample.png")

# Combined side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(Image.open(normal_img_path), cmap="gray")
axes[0].set_title("NORMAL")
axes[0].axis("off")

axes[1].imshow(Image.open(pneumonia_img_path), cmap="gray")
axes[1].set_title("PNEUMONIA")
axes[1].axis("off")

plt.suptitle("Sample Chest X-Ray Images from Test Dataset")
plt.tight_layout()
plt.savefig("dataset_samples.png", dpi=300)
plt.show()
