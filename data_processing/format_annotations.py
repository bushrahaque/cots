import os
import pandas as pd
import cv2

# RELEVANT DIRECTORY PATHS
LABELS_CSV_PATH = '/Users/bushra/Documents/STA2453/cots/data/all_labels.csv'
IMAGES_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/images'
LABELS_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/labels'


def normalize_bbox(bbox, image_width, image_height):
    """
    Normalizes a bounding box to YOLO format using image dimensions.

    Inputs:
    - bbox (dict): Dictionary containing "x", "y", "width", and "height" keys. Values are integers.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Outputs:
    - str: A string formatted as 'class x_center y_center width height' (YOLO format).
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w /= image_width
    h /= image_height

    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def convert_annotations(labels_df, orig_images_dir, labels_dir):
    """
    Converts raw annotation data from a DataFrame into YOLO-formatted label files.
    The annotation strings have format:
        "[{'x': 0, 'y': 0, 'width': 0, 'height': 0}, {'x': 1, 'y': 1, 'width': 1, 'height': 1}]"

    Inputs:
    - labels_df (pd.DataFrame): DataFrame containing image IDs and annotation strings.
    - orig_images_dir (str): Path to the directory containing images.
    - labels_dir (str): Path to the directory where YOLO label text files will be saved.

    Outputs:
    - None
    """
    for _, row in labels_df.iterrows():
        image_id = row["image_id"]
        image_path = os.path.join(orig_images_dir, f"{image_id}.jpg")
        label_path = os.path.join(labels_dir, f"{image_id}.txt")

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_id} not found, skipping.")
            continue

        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_id}, skipping.")
            continue

        image_height, image_width = image.shape[:2]

        # Skip if no annotations or empty list
        if pd.isna(row["annotations"]) or row["annotations"] == "[]":
            continue

        # Parse annotation string into a list of bounding boxes
        annotations = eval(row["annotations"])

        # Write each normalized bbox to the corresponding text file
        with open(label_path, "w") as f:
            for ann in annotations:
                f.write(normalize_bbox(ann, image_width, image_height) + "\n")

    print(f"Success! The formatted annotation text files have been saved in: {labels_dir}")


if __name__ == "__main__":

    # Verify output directory exists
    os.makedirs(LABELS_DIR, exist_ok=True)

    # Load annotation data
    labels_df = pd.read_csv(LABELS_CSV_PATH)

    # Convert annotations to YOLO format
    convert_annotations(
        labels_df=labels_df,
        orig_images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
    )
