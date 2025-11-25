import matplotlib.pyplot as plt
import numpy as np

# Average times per image (seconds) from your experiments
labels = ["Plain FC1", "Encryption", "Encrypted FC1"]
times = [0.006, 0.063, 29.383]

x = np.arange(len(labels))

plt.figure(figsize=(6, 4))
plt.bar(x, times)
plt.xticks(x, labels, rotation=15)
plt.ylabel("Time per image (seconds)")
plt.title("Runtime Comparison: Plain vs Encrypted Inference")

# Optional: log-scale y-axis to show differences more clearly
plt.yscale("log")
plt.tight_layout()
plt.savefig("runtime_bar.png", dpi=300)
plt.show()
