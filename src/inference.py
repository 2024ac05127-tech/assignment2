from pathlib import Path
import io
import logging
import time

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
# Monitoring metrics
# ============================================================

request_count = 0
prediction_count = 0
error_count = 0
total_latency = 0.0


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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
# Metrics endpoint
# ============================================================

@app.get("/metrics")
def metrics():

    average_latency = (
        total_latency / prediction_count
        if prediction_count > 0
        else 0.0
    )

    return {
        "request_count": request_count,
        "prediction_count": prediction_count,
        "error_count": error_count,
        "average_latency_seconds": round(
            average_latency,
            4
        )
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    global request_count
    global prediction_count
    global error_count
    global total_latency

    request_count += 1

    start_time = time.time()

    logger.info(
        "Prediction request received"
    )

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:

        error_count += 1

        logger.warning(
            "Unsupported image format: %s",
            file.content_type
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG or WEBP."
            )
        )

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_label = (
            CLASS_NAMES[predicted_index]
        )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probability_dict = {

            CLASS_NAMES[i]: round(
                float(probabilities[i]),
                4
            )

            for i in range(
                len(CLASS_NAMES)
            )
        }

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency = time.time() - start_time

        prediction_count += 1

        total_latency += latency

        logger.info(
            "Prediction completed: label=%s latency=%.4fs",
            predicted_label,
            latency
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "filename": file.filename,

            "label": predicted_label,

            "probabilities": probability_dict

        }

    except Exception as exc:

        error_count += 1

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )