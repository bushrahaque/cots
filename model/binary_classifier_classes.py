import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
import cv2
import glob

# Image transformation for CNN input
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class COTSDataset(Dataset):
    """
    Custom PyTorch dataset for loading underwater images and binary COTS labels.

    An image is labeled as 1 (COTS present) if a corresponding YOLO label file exists,
    otherwise it is labeled as 0 (no COTS present).

    Inputs:
    - image_dir (str): Directory containing image files.
    - label_dir (str): Directory containing YOLO-style label .txt files.
    - transform (callable, optional): Optional transform to be applied on a sample.

    Outputs:
    - A tuple (image tensor, label) for each image.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        # Binary label based on file presence
        label = 1 if os.path.exists(label_path) else 0  

        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


class COTSClassifier(nn.Module):
    """
    CNN-based binary classifier using ResNet-18 for COTS detection.

    The final fully connected layer is modified for two output classes:
    - Class 0: No COTS
    - Class 1: COTS present
    """
    def __init__(self):
        super(COTSClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)
