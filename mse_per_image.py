import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("encrypted_eval_summary.csv")

# Drop rows where idx is NOT a number (e.g., 'AVERAGE')
df_clean = df[pd.to_numeric(df["idx"], errors="coerce").notnull()].copy()

# Convert idx to integer
df_clean["idx"] = df_clean["idx"].astype(int)

# --- MSE per image ---
plt.figure(figsize=(6, 4))
plt.bar(df_clean["idx"], df_clean["MSE"])
plt.xlabel("Image index")
plt.ylabel("MSE")
plt.title("MSE per Image (Encrypted vs Plain FC1)")
plt.xticks(df_clean["idx"])
plt.tight_layout()
plt.savefig("mse_per_image.png", dpi=300)
plt.show()

# --- Correlation per image ---
plt.figure(figsize=(6, 4))
plt.bar(df_clean["idx"], df_clean["Corr"])
plt.xlabel("Image index")
plt.ylabel("Correlation")
plt.ylim(0.99, 1.0)
plt.title("Correlation per Image (Encrypted vs Plain FC1)")
plt.xticks(df_clean["idx"])
plt.tight_layout()
plt.savefig("corr_per_image.png", dpi=300)
plt.show()
