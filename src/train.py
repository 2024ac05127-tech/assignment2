import os

import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch

import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ==================================================
# Configuration
# ==================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 0.001

NUM_CLASSES = 2

SEED = 42

torch.manual_seed(SEED)


# ==================================================
# Directories
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "artifacts",
    exist_ok=True
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# Data Augmentation
# ==================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# Load datasets
# ==================================================

train_dataset = datasets.ImageFolder(
    "data/processed/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "data/processed/val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    "data/processed/test",
    transform=val_test_transform
)


print(
    "Classes:",
    train_dataset.classes
)


# ==================================================
# Data loaders
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# Baseline CNN
# ==================================================

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 224 -> 112
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 112 -> 56
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 56 -> 28
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 28 -> 14
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 14 * 14,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = BaselineCNN().to(device)


# ==================================================
# Loss and optimizer
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# MLflow
# ==================================================

mlflow.set_experiment(
    "Cats_Dogs_Baseline_CNN"
)


with mlflow.start_run():

    # ----------------------------------------------
    # Log parameters
    # ----------------------------------------------

    mlflow.log_param(
        "model",
        "BaselineCNN"
    )

    mlflow.log_param(
        "image_size",
        IMAGE_SIZE
    )

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE
    )

    mlflow.log_param(
        "epochs",
        EPOCHS
    )

    mlflow.log_param(
        "learning_rate",
        LEARNING_RATE
    )

    mlflow.log_param(
        "optimizer",
        "Adam"
    )

    mlflow.log_param(
        "augmentation",
        "HorizontalFlip, Rotation, ColorJitter"
    )


    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    train_losses = []

    val_losses = []

    train_accuracies = []

    val_accuracies = []


    for epoch in range(EPOCHS):

        # ==========================================
        # TRAIN
        # ==========================================

        model.train()

        running_loss = 0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()

            optimizer.step()


            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        train_loss = (
            running_loss
            / len(train_dataset)
        )

        train_accuracy = (
            correct / total
        )


        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()

        val_loss = 0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                val_loss += (
                    loss.item()
                    * images.size(0)
                )


                _, predicted = torch.max(
                    outputs,
                    1
                )


                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()


        val_loss = (
            val_loss
            / len(val_dataset)
        )

        val_accuracy = (
            val_correct
            / val_total
        )


        # Store values

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )


        # MLflow

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch
        )

        mlflow.log_metric(
            "train_accuracy",
            train_accuracy,
            step=epoch
        )

        mlflow.log_metric(
            "val_accuracy",
            val_accuracy,
            step=epoch
        )


        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


    # ==================================================
    # TEST
    # ==================================================

    model.eval()

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )


    # ==================================================
    # Test accuracy
    # ==================================================

    correct = sum(
        p == y
        for p, y in zip(
            all_predictions,
            all_labels
        )
    )

    test_accuracy = (
        correct
        / len(all_labels)
    )


    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )


    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )


    # ==================================================
    # Confusion Matrix
    # ==================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )


    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        "Cats vs Dogs Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = (
        "artifacts/confusion_matrix.png"
    )

    plt.savefig(
        confusion_path
    )

    plt.close()


    mlflow.log_artifact(
        confusion_path
    )


    # ==================================================
    # Loss Curve
    # ==================================================

    plt.figure()

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.tight_layout()

    loss_path = (
        "artifacts/loss_curve.png"
    )

    plt.savefig(
        loss_path
    )

    plt.close()


    mlflow.log_artifact(
        loss_path
    )


    # ==================================================
    # Accuracy Curve
    # ==================================================

    plt.figure()

    plt.plot(
        train_accuracies,
        label="Train Accuracy"
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_path = (
        "artifacts/accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path
    )

    plt.close()


    mlflow.log_artifact(
        accuracy_path
    )


    # ==================================================
    # Save model
    # ==================================================

    model_path = (
        "models/baseline_cnn.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    # Log model

    mlflow.log_artifact(
        model_path
    )


    mlflow.pytorch.log_model(
        model,
        "model"
    )


    print(
        "Model and artifacts saved."
    )