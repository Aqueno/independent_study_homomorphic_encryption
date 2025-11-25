import os

# Update this path with your username
base_path = r"C:\Users\niras\independent_study\data\chest_xray"

for split in ["train", "val", "test"]:
    path = os.path.join(base_path, split)
    if os.path.exists(path):
        count = sum([len(files) for r, d, files in os.walk(path)])
        print(f"{split} : {count} images")
    else:
        print(f"{split} folder not found at {path}")
