import os
import pandas as pd
import shutil

# EXISTING DATA DIRS
LABELS_CSV_PATH = '/Users/bushra/Documents/STA2453/cots/data/all_labels.csv'

IMAGES_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/images'
LABELS_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/labels'

# FORMATTED DATA DIRS
DATA_DIR = '/Users/bushra/Documents/STA2453/cots/data/'


def split_data(labels_df, split_images_dir, split_labels_dir, orig_images_dir, orig_labels_dir):
    """
    Copies image and label files from the original dataset directories to the specified split directories
    based on the entries in the provided DataFrame.

    Inputs:
    - labels_df (pd.DataFrame): DataFrame containing the image_id per frame in each video.
    - split_images_dir (str): Path to the destination directory for the image split (e.g., train/images).
    - split_labels_dir (str): Path to the destination directory for the label split (e.g., train/labels).
    - orig_images_dir (str): Path to the original directory containing image files.
    - orig_labels_dir (str): Path to the original directory containing label files.

    Outputs:
    - None
    """
    for _, row in labels_df.iterrows():
        # Construct full paths to the original image and label files
        image_path = os.path.join(orig_images_dir, f"{row['image_id']}.jpg")
        label_path = os.path.join(orig_labels_dir, f"{row['image_id']}.txt")

        # Copy files if the image exists
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(split_images_dir, f"{row['image_id']}.jpg"))

            # Copy label file only if it exists
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(split_labels_dir, f"{row['image_id']}.txt"))
        else:
            print(f"Missing image file for {row['image_id']}")

    return None


if __name__ == "__main__":
    
    # Create the directory structure for train, val, and test splits
    for split in ['train', 'val', 'test']:
        for _type in ['images', 'labels']:
            os.makedirs(f'/Users/bushra/Documents/STA2453/cots/data/{split}/{_type}', exist_ok=True)

    # Read the CSV file that contains split information
    labels_df = pd.read_csv(LABELS_CSV_PATH)

    # Map each video ID to the appropriate data split
    for v, split in zip([0, 1, 2], ['train', 'test', 'val']):
        SPLIT_IMAGES_DIR = os.path.join(DATA_DIR, f"{split}/images")
        SPLIT_LABELS_DIR = os.path.join(DATA_DIR, f"{split}/labels")

        # Call the function to copy the files into the right folders
        split_data(
            labels_df.loc[labels_df["video_id"] == v],
            SPLIT_IMAGES_DIR,
            SPLIT_LABELS_DIR,
            IMAGES_DIR,
            LABELS_DIR
        )

