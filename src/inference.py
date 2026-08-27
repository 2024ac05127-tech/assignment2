from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

CLASS_NAMES = [
    "cats",
    "dogs",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_cnn.pt"
)


# ============================================================
# Model definition
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

            # IMPORTANT:
            # This must be present because
            # the trained model contains classifier.4
            nn.Dropout(
                p=0.5
            ),

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
# Device
# ============================================================

device = torch.device("cpu")


# ============================================================
# Load model
# ============================================================

model = BaselineCNN()

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


# ============================================================
# Image preprocessing
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


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description=(
        "REST API for binary image classification "
        "using a baseline CNN."
    ),
    version="1.0.0"
)


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "BaselineCNN",
        "model_file": MODEL_PATH.name,
        "device": str(device)
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Validate file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG or WEBP."
            )
        )


    try:

        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")


        # Preprocess
        image_tensor = transform(
            image
        )

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)


        # Prediction
        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]


        # Predicted class
        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_label = (
            CLASS_NAMES[predicted_index]
        )


        # Convert probabilities
        probability_dict = {

            CLASS_NAMES[i]: round(
                float(probabilities[i]),
                4
            )

            for i in range(
                len(CLASS_NAMES)
            )
        }


        return {

            "filename": file.filename,

            "label": predicted_label,

            "probabilities": probability_dict

        }


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )