import os, random, time, glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import tenseal as ts
from week2_cnn_train_improved import ImprovedCNN  # your model

# ---- config ----
DATA_ROOT = r"C:\Users\niras\Desktop\independent_study\data\chest_xray\test"
CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGES_PER_CLASS = 3            # tweak (3–5)
POLY_MOD_DEGREE = 16384         # good balance
COEFF_MOD = [40, 21, 21, 40]
GLOBAL_SCALE = 2**21

# ---- setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load("week2_cnn_improved.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# build image list
def pick_images():
    paths = []
    for cls in CLASSES:
        folder = os.path.join(DATA_ROOT, cls)
        cand = glob.glob(os.path.join(folder, "*"))
        random.shuffle(cand)
        paths += [(p, cls) for p in cand[:IMAGES_PER_CLASS]]
    random.shuffle(paths)
    return paths

def load_features(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.conv(x).view(1, -1).cpu().numpy()[0]   # (50176,)
    return feats

# encrypted FC1 (multi-ciphertext)
def enc_fc1(features, fc1_w, fc1_b, poly_deg=POLY_MOD_DEGREE):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_deg, -1, COEFF_MOD)
    ctx.generate_galois_keys()
    ctx.global_scale = GLOBAL_SCALE

    max_slots = poly_deg // 2
    chunks = [features[i:i+max_slots] for i in range(0, len(features), max_slots)]

    t_enc0 = time.time()
    enc_chunks = [ts.ckks_vector(ctx, c.tolist()) for c in chunks]
    enc_time = time.time() - t_enc0

    t_inf0 = time.time()
    outs = []
    for i in range(fc1_w.shape[0]):   # 128 neurons
        total = 0.0
        for j, c in enumerate(chunks):
            wj = fc1_w[i, j*max_slots : j*max_slots + len(c)]
            if len(wj) == 0: continue
            total += enc_chunks[j].dot(wj.tolist()).decrypt()[0]
        outs.append(total + float(fc1_b[i]))
    inf_time = time.time() - t_inf0
    return np.array(outs), enc_time, inf_time

# weights
fc1_w = model.fc[0].weight.detach().cpu().numpy()  # (128,50176)
fc1_b = model.fc[0].bias.detach().cpu().numpy()    # (128,)

# run
rows = []
samples = pick_images()
print(f"Running on {len(samples)} images: {IMAGES_PER_CLASS} per class")

for idx, (img_path, label) in enumerate(samples, 1):
    feats = load_features(img_path)

    # plain FC1
    t0 = time.time()
    plain = np.dot(fc1_w, feats) + fc1_b
    plain_t = time.time() - t0

    # encrypted FC1 (full features via multi-ciphertext)
    enc_out, t_enc, t_inf = enc_fc1(feats, fc1_w, fc1_b, POLY_MOD_DEGREE)

    mse = np.mean((enc_out - plain)**2)
    corr = np.corrcoef(enc_out, plain)[0,1]

    rows.append((idx, os.path.basename(img_path), label, round(mse,3),
                 round(corr,6), round(plain_t,3), round(t_enc,3), round(t_inf,3)))

# print table
print("\nResults (per image)")
print("idx | file | class | MSE | Corr | Plain(s) | Encrypt(s) | Enc+Infer(s)")
for r in rows:
    print(f"{r[0]:>3} | {r[1][:20]:<20} | {r[2]:<9} | {r[3]:>6} | {r[4]:>6} | {r[5]:>8} | {r[6]:>10} | {r[7]:>12}")

# quick summary
avg_mse = round(np.mean([r[3] for r in rows]),3)
avg_corr = round(np.mean([r[4] for r in rows]),6)
avg_plain = round(np.mean([r[5] for r in rows]),3)
avg_enc = round(np.mean([r[6] for r in rows]),3)
avg_inf = round(np.mean([r[7] for r in rows]),3)

print("\nAverages across images:")
print(f"MSE={avg_mse}, Corr={avg_corr}, Plain={avg_plain}s, Encrypt={avg_enc}s, Enc+Infer={avg_inf}s")

# optional: save CSV
try:
    import csv
    out_csv = "encrypted_eval_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","file","class","MSE","Corr","Plain_s","Encrypt_s","EncInfer_s"])
        for r in rows: w.writerow(r)
        w.writerow([])
        w.writerow(["AVERAGE","","",avg_mse,avg_corr,avg_plain,avg_enc,avg_inf])
    print(f"\nSaved CSV: {out_csv}")
except Exception as e:
    print("CSV save skipped:", e)
