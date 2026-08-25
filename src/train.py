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


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 0.001

NUM_CLASSES = 2

SEED = 42

torch.manual_seed(SEED)


# ============================================================
# Directories
# ============================================================

MODEL_DIR = "models"

ARTIFACT_DIR = "artifacts"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 60)
print("Cats vs Dogs - Baseline CNN")
print("=" * 60)

print(
    f"Using device: {device}"
)


# ============================================================
# Data transformations
# ============================================================

# Training:
# Data augmentation is applied ONLY to training data.

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Validation and test:
# NO augmentation.

val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Load datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    root="data/processed/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root="data/processed/val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root="data/processed/test",
    transform=val_test_transform
)


# ============================================================
# IMPORTANT: Class mapping
# ============================================================

print()
print("Classes:")
print(train_dataset.classes)

print()
print("Class mapping:")
print(train_dataset.class_to_idx)


# Expected:
#
# Classes:
# ['cats', 'dogs']
#
# Class mapping:
# {'cats': 0, 'dogs': 1}


# ============================================================
# Verify class mapping
# ============================================================

if train_dataset.class_to_idx != {
    "cats": 0,
    "dogs": 1
}:

    raise ValueError(
        "Unexpected class mapping. "
        f"Found: {train_dataset.class_to_idx}"
    )


# ============================================================
# Data loaders
# ============================================================

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


print()
print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)


# ============================================================
# Baseline CNN
# ============================================================

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 224 x 224
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 112 x 112
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 56 x 56
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 28 x 28
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)

            # Output:
            # 128 x 14 x 14
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 14 * 14,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.5
            ),

            # 2 outputs:
            # index 0 = cats
            # index 1 = dogs
            nn.Linear(
                128,
                NUM_CLASSES
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Create model
# ============================================================

model = BaselineCNN()

model = model.to(device)


print()
print("Model created.")


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# MLflow
# ============================================================

mlflow.set_experiment(
    "Cats_Dogs_Baseline_CNN"
)


with mlflow.start_run():

    print()
    print("MLflow run started.")


    # ========================================================
    # Log parameters
    # ========================================================

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
        "num_classes",
        NUM_CLASSES
    )

    mlflow.log_param(
        "augmentation",
        "HorizontalFlip + Rotation + ColorJitter"
    )


    # ========================================================
    # Lists for plots
    # ========================================================

    train_losses = []

    val_losses = []

    train_accuracies = []

    val_accuracies = []


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # Clear gradients
            optimizer.zero_grad()


            # Forward pass
            outputs = model(images)


            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            # Statistics
            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predictions = torch.max(
                outputs,
                dim=1
            )


            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


        train_loss = (
            running_loss
            / len(train_dataset)
        )

        train_accuracy = (
            correct / total
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0

        validation_correct = 0

        validation_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                validation_loss += (
                    loss.item()
                    * images.size(0)
                )


                _, predictions = torch.max(
                    outputs,
                    dim=1
                )


                validation_total += (
                    labels.size(0)
                )


                validation_correct += (
                    predictions == labels
                ).sum().item()


        val_loss = (
            validation_loss
            / len(val_dataset)
        )

        val_accuracy = (
            validation_correct
            / validation_total
        )


        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MLflow metrics
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)


            _, predictions = torch.max(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


    # ========================================================
    # Test accuracy
    # ========================================================

    test_correct = sum(

        prediction == label

        for prediction, label

        in zip(
            all_predictions,
            all_labels
        )
    )


    test_accuracy = (
        test_correct
        / len(all_labels)
    )


    print()
    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )


    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )


    # ========================================================
    # Classification report
    # ========================================================

    report = classification_report(

        all_labels,

        all_predictions,

        target_names=[
            "cats",
            "dogs"
        ]
    )


    print()
    print("Classification Report:")
    print(report)


    # ========================================================
    # Confusion Matrix
    # ========================================================

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

        xticklabels=[
            "cats",
            "dogs"
        ],

        yticklabels=[
            "cats",
            "dogs"
        ]
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Cats vs Dogs Confusion Matrix"
    )


    plt.tight_layout()


    confusion_matrix_path = (
        f"{ARTIFACT_DIR}/confusion_matrix.png"
    )


    plt.savefig(
        confusion_matrix_path
    )

    plt.close()


    mlflow.log_artifact(
        confusion_matrix_path
    )


    # ========================================================
    # Loss curve
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, EPOCHS + 1),
        train_losses,
        label="Train Loss"
    )


    plt.plot(
        range(1, EPOCHS + 1),
        val_losses,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()


    loss_curve_path = (
        f"{ARTIFACT_DIR}/loss_curve.png"
    )


    plt.savefig(
        loss_curve_path
    )

    plt.close()


    mlflow.log_artifact(
        loss_curve_path
    )


    # ========================================================
    # Accuracy curve
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, EPOCHS + 1),
        train_accuracies,
        label="Train Accuracy"
    )


    plt.plot(
        range(1, EPOCHS + 1),
        val_accuracies,
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()


    accuracy_curve_path = (
        f"{ARTIFACT_DIR}/accuracy_curve.png"
    )


    plt.savefig(
        accuracy_curve_path
    )

    plt.close()


    mlflow.log_artifact(
        accuracy_curve_path
    )


    # ========================================================
    # Save model
    # ========================================================

    model_path = (
        f"{MODEL_DIR}/baseline_cnn.pt"
    )


    torch.save(
        model.state_dict(),
        model_path
    )


    print()
    print(
        f"Model saved to: "
        f"{model_path}"
    )


    # ========================================================
    # Log model to MLflow
    # ========================================================

    mlflow.log_artifact(
        "mlops-project/models/baseline_cnn.pt"
    )


    mlflow.pytorch.log_model(
        model,
        "model"
    )

    example_images, _ = next(iter(test_loader))

    example_images = example_images.to(device)

    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example_images[:1]
    )
    print()
    print("=" * 60)
    print("MLflow run completed")
    print("=" * 60)