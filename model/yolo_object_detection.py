from ultralytics import YOLO
import torch

# RELEVANT PATHS
LABELS_CSV_PATH = '/Users/bushra/Documents/STA2453/cots/data/all_labels.csv'
YOLO_DATASET_YAML = "/Users/bushra/Documents/STA2453/cots/model/yolo_data_config.yaml"
YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/yolov8n.pt"
TUNED_YOLO_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/yolov8n_tuned.pt"

def tune_yolo(epochs=100, batch=16, imgsz=640, augment=True):
    """
    Tunes YOLOv8n on the COTS dataset with custom training parameters.
    
    Inputs:
    - epochs (int): Number of training epochs.
    - batch (int): Batch size.
    - imgsz (int): Input image size.
    - augment (bool): Whether to apply data augmentation.

    Outputs:
    - None
    """
    print("Starting YOLO Hyperparameter Tuning...")

    # Load the base YOLOv8 model
    model = YOLO(YOLO_MODEL_PATH)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    model.train(
        data=YOLO_DATASET_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        augment=augment,
        save=True,
        save_period=1,
        val=True
    )

    # Save the tuned model
    model.save(TUNED_YOLO_MODEL_PATH)
    print(f"YOLO tuning complete! Tuned model saved at {TUNED_YOLO_MODEL_PATH}")

if __name__ == "__main__":
    tune_yolo(
        epochs=10,
        batch=64,
        imgsz=1280
    )

