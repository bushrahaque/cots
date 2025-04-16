import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from cots.model.binary_classifier_classes import transform, COTSDataset, COTSClassifier

# PATHS
DATA_DIR = "/Users/bushra/Documents/STA2453/cots/data/"
CLASSIFER_MODEL_PATH = "/Users/bushra/Documents/STA2453/cots/model/outputs/binary_classifier.pth"


def train_cots_classifier(num_epochs=1, batch_size=16, learning_rate=0.01):
    """
    Trains a CNN-based binary classifier to detect the presence of COTS in underwater images.
    The model has been pre-defined in the global_definitions.py module.

    Inputs:
    - num_epochs (int): Number of epochs to train the model.
    - batch_size (int): Number of samples per training batch.
    - learning_rate (float): Learning rate for the optimizer.

    Outputs:
    - None
    """
    print("Starting CNN Training for COTS Detection...")

    # Prepare training dataset
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train/images")
    TRAIN_LABELS_DIR = os.path.join(DATA_DIR, "train/labels")
    train_dataset = COTSDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation dataset
    VAL_IMAGES_DIR = os.path.join(DATA_DIR, "val/images")
    VAL_LABELS_DIR = os.path.join(DATA_DIR, "val/labels")
    val_dataset = COTSDataset(VAL_IMAGES_DIR, VAL_LABELS_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COTSClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('Inf')

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress", leave=True):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CLASSIFER_MODEL_PATH)
            print(f"New best model saved to {CLASSIFER_MODEL_PATH}")


if __name__ == "__main__":
    train_cots_classifier()

