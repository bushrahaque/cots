import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Define paths
TRAIN_CSV_PATH = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train.csv"
IMAGE_DIR = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/images"
LABEL_DIR = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/labels"
OUTPUT_DIR = "/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/"

# Create YOLO directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, f"labels/{split}"), exist_ok=True)

def copy_files(df, split):
    for _, row in df.iterrows():
        image_path = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
        label_path = os.path.join(LABEL_DIR, f"{row['image_id']}.txt")
        
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(OUTPUT_DIR, f"images/{split}/{row['image_id']}.jpg"))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(OUTPUT_DIR, f"labels/{split}/{row['image_id']}.txt"))
        else:
            print(f"Missing image file for {row['image_id']}")

def split_dataset(test_size=0.2, val_size=0.1, random_state=2453):
    # Load full dataset
    df = pd.read_csv(TRAIN_CSV_PATH)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Further split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=random_state)

    # Move files to YOLO format
    copy_files(train_df, 'train')
    copy_files(val_df, 'val')
    copy_files(test_df, 'test')

    print("Train/Val/Test split completed.")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

if __name__ == "__main__":
    split_dataset()

