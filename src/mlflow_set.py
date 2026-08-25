import torch
import mlflow
import mlflow.pytorch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 1
MODEL_PATH = "models/baseline_cnn.pt"


# ============================================================
# Model definition
# ============================================================

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
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

            nn.Dropout(p=0.5),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# Load model
# ============================================================

model = BaselineCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Create example input
# ============================================================

transform = transforms.Compose([

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


test_dataset = datasets.ImageFolder(
    root="data/processed/test",
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


example_images, _ = next(iter(test_loader))


# ============================================================
# Log model to MLflow
# ============================================================

mlflow.set_experiment(
    "Cats_Dogs_Baseline_CNN"
)

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "BaselineCNN"
    )

    mlflow.log_param(
        "image_size",
        IMAGE_SIZE
    )

    mlflow.log_param(
        "num_classes",
        2
    )

    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example_images,
        serialization_format="pickle"
    )

    print("Model successfully logged to MLflow.")