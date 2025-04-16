import os
import pandas as pd
import torch
import cv2
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import DataLoader
from cots.model.binary_classifier_classes import COTSClassifier, COTSDataset, transform
from ultralytics import YOLO

# RELEVANT PATHS
LABELS_CSV_PATH = '/Users/bushra/Documents/STA2453/cots/data/all_labels.csv'
DATA_DIR = "/Users/bushra/Documents/STA2453/cots/data/"

YOLO_DATASET_YAML = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"
TUNED_YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/yolov8n_tuned.pt"
CLASSIFIER_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/binary_classifier.pth"



def evaluate_binary_classifier(batch_size=16):
    """
    Evaluate the performance of the binary COTS classifier on the test set. Prints results.

    Inputs:
    - batch_size (int): Batch size for DataLoader.

    Ouputs:
    - None
    """
    print("Evaluating binary classifier...")

    # Load test dataset
    test_df = pd.read_csv(LABELS_CSV_PATH)
    test_df = test_df.loc[test_df['video_id']==2]

    # Prepare test dataset
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test/images")
    TEST_LABELS_DIR = os.path.join(DATA_DIR, "test/labels")
    test_dataset = COTSDataset(TEST_IMAGES_DIR, TEST_LABELS_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load trained classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COTSClassifier().to(device)
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # Perform inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=['No COTS', 'COTS'])
    print("Classification Report (Binary Classifier):\n", report)


def evaluate_ensemble(use_classifier=True):
    """
    Evaluate the full pipeline (binary classifier + YOLO) on the test set.

    If use_classifier is True, only images predicted as positive by the classifier
    are passed to YOLO. Otherwise, YOLO runs on the full test set.

    Inputs:
    - use_classifier (bool): Whether to filter test images using the binary classifier.

    Outputs:
    - dict: YOLO evaluation metrics including mAP, precision, and recall.
    """
    print(f"Evaluating ensemble model (use_classifier={use_classifier})...")

    # Load test image paths
    test_df = pd.read_csv(LABELS_CSV_PATH)
    test_df = test_df.loc[test_df['video_id']==2]
    image_paths = [os.join.path(DATA_DIR, f"test/images/{image_id}.jpg") for image_id in test_df['image_id'].tolist()]

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = COTSClassifier().to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO(TUNED_YOLO_MODEL_PATH)

    # Filter images through binary classifier (if applicable)
    selected_images = []
    for path in image_paths:
        if not use_classifier:
            selected_images.append(path)
        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier(image)
                pred = torch.argmax(output, dim=1).item()
                if pred == 1:
                    selected_images.append(path)

    if not selected_images:
        print("No images selected for YOLO evaluation.")
        return {}

    # Evaluate YOLO
    metrics = yolo_model.val(data=YOLO_DATASET_YAML, imgsz=1280, batch=64, device=device)
    print("YOLO mAP Evaluation Metrics:\n", metrics)


if __name__ == "__main__":
    evaluate_binary_classifier(batch_size=64)
    evaluate_ensemble(use_classifier=True)
    
