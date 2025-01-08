import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description="Split data into train-validation-test sets.")

parser.add_argument("--other", action="store_true", help="Process other data mode (default is process carpets).")
args = parser.parse_args()

dir = "carpets" if not args.other else "other"

SOURCE_DIR = f"datasets/{dir}"

TRAIN_DIR = f"datasets/train/{dir}"
VAL_DIR   = f"datasets/val/{dir}"
TEST_DIR  = f"datasets/test/{dir}"

os.makedirs(TRAIN_DIR, exist_ok=False)
os.makedirs(VAL_DIR, exist_ok=False)
os.makedirs(TEST_DIR, exist_ok=False)

# List all image files in SOURCE_DIR
all_files = [
    f for f in os.listdir(SOURCE_DIR) 
    if os.path.isfile(os.path.join(SOURCE_DIR, f))
    and (f.lower().endswith(".jpg") or f.lower().endswith(".png"))
]

random.shuffle(all_files) # Shuffle the file list for randomness

# define split sizes (here it s 70% train, 15% val, 15% test)
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

total_count = len(all_files)
train_count = int(train_ratio * total_count)
val_count   = int(val_ratio * total_count)


train_files = all_files[:train_count]
val_files   = all_files[train_count : train_count + val_count]
test_files  = all_files[train_count + val_count :]

print(f"Total images: {total_count}")
print(f"Training: {len(train_files)}")
print(f"Validation: {len(val_files)}")
print(f"Test: {len(test_files)}")

# Copy files to each directory
for f in train_files:
    shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(TRAIN_DIR, f))

for f in val_files:
    shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(VAL_DIR, f))

for f in test_files:
    shutil.copy(os.path.join(SOURCE_DIR, f), os.path.join(TEST_DIR, f))

print("Done splitting!")
