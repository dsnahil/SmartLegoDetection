import os
import random
import shutil
import xml.etree.ElementTree as ET

# -----------------------
# Configuration Variables
# -----------------------
FULL_DATASET_PATH = 'full'
REDUCED_DATASET_PATH = 'reduced'
IMAGES_DIR = os.path.join(FULL_DATASET_PATH, 'images')
ANNOTATIONS_DIR = os.path.join(FULL_DATASET_PATH, 'annotations')
SAMPLE_SIZE = 10000  # Total number of images to keep

# -----------------------
# Helper Functions
# -----------------------
def is_valid_annotation(xml_file):
    """
    Check if the annotation file is valid.

    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = root.findall('object')
        return len(objects) > 0             #If single or more objects are present then it is okay
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return False

def create_folder_structure(base_dir):
    """
    Create folders for train, val, and test splits.
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'annotations'), exist_ok=True)
    print(f"Folder structure created in '{base_dir}'")

# -----------------------
# Main Process
# -----------------------

print("Starting dataset reduction process...")

# Step 1: Create the reduced dataset folder structure
create_folder_structure(REDUCED_DATASET_PATH)

# Step 2: Gather valid image-annotation pairs
print("Collecting valid annotation files...")
valid_files = []
ann_files = os.listdir(ANNOTATIONS_DIR)
random.shuffle(ann_files)  # Shuffle to ensure randomness

for idx, ann_file in enumerate(ann_files):
    ann_path = os.path.join(ANNOTATIONS_DIR, ann_file)
    if is_valid_annotation(ann_path):
        base_name = os.path.splitext(ann_file)[0]
        image_file = base_name + '.jpg'
        image_path = os.path.join(IMAGES_DIR, image_file)
        if os.path.exists(image_path):
            valid_files.append((image_file, ann_file))
            # Stop if we've reached our sample size
            if len(valid_files) >= SAMPLE_SIZE:
                print(f"Reached sample size: {SAMPLE_SIZE} valid files collected.")
                break
    # Print progress every 1000 processed annotation files
    if idx % 1000 == 0 and idx > 0:
        print(f"Processed {idx} annotation files. Current valid files count: {len(valid_files)}")

print(f"Total valid files collected: {len(valid_files)}")

# Ensure SAMPLE_SIZE does not exceed available valid files
if len(valid_files) < SAMPLE_SIZE:
    SAMPLE_SIZE = len(valid_files)
    print(f"Adjusted SAMPLE_SIZE to {SAMPLE_SIZE} due to limited valid files.")

# Step 3: Randomly sample a subset of valid files
print("Sampling valid files...")
sampled_files = random.sample(valid_files, SAMPLE_SIZE)

# Step 4: Split the data into train, val, test splits
train_count = int(SAMPLE_SIZE * 0.7)
val_count = int(SAMPLE_SIZE * 0.15)
train_files = sampled_files[:train_count]
val_files = sampled_files[train_count:train_count + val_count]
test_files = sampled_files[train_count + val_count:]

print(f"Splitting data: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")

# Step 5: Copy the files to their respective folders
def copy_files(file_list, split):
    total_files = len(file_list)
    print(f"Copying {total_files} files to the '{split}' folder...")
    for i, (img_file, ann_file) in enumerate(file_list, 1):
        src_img = os.path.join(IMAGES_DIR, img_file)
        src_ann = os.path.join(ANNOTATIONS_DIR, ann_file)
        dst_img = os.path.join(REDUCED_DATASET_PATH, split, 'images', img_file)
        dst_ann = os.path.join(REDUCED_DATASET_PATH, split, 'annotations', ann_file)
        shutil.copy(src_img, dst_img)
        shutil.copy(src_ann, dst_ann)
        if i % 500 == 0 or i == total_files:
            print(f"{i}/{total_files} files copied for '{split}' split.")

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset reduction and splitting complete.")