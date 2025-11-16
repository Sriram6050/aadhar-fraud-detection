import os
import csv

DATASET_PATH = "datasets"
OUTPUT_FILE = "metadata.csv"

rows = []
for folder in ["genuine", "tampered"]:
    folder_path = os.path.join(DATASET_PATH, folder)
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([os.path.join(folder, file), folder])

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print("✅ metadata.csv created successfully!")
