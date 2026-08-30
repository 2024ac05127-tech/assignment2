import sys
import requests

from PIL import Image
import io


BASE_URL = "http://localhost:8000"


# ============================================================
# Health Check
# ============================================================

print("Testing health endpoint...")

response = requests.get(
    f"{BASE_URL}/health",
    timeout=10
)

if response.status_code != 200:

    print("Health check FAILED")
    print(response.text)

    sys.exit(1)


health_result = response.json()

print("Health check PASSED")
print(health_result)


# ============================================================
# Prediction Test
# ============================================================

print()
print("Testing prediction endpoint...")


# Create dummy JPEG image

image = Image.new(
    "RGB",
    (224, 224),
    color="white"
)

image_bytes = io.BytesIO()

image.save(
    image_bytes,
    format="JPEG"
)

image_bytes.seek(0)


files = {
    "file": (
        "test.jpg",
        image_bytes,
        "image/jpeg"
    )
}


response = requests.post(
    f"{BASE_URL}/predict",
    files=files,
    timeout=30
)


# Check HTTP response

if response.status_code != 200:

    print("Prediction test FAILED")

    print(response.text)

    sys.exit(1)


result = response.json()

print("Prediction request PASSED")
print(result)


# ============================================================
# Validate Prediction
# ============================================================

if "label" not in result:

    print("Label field missing")

    sys.exit(1)


if result["label"] not in [
    "cats",
    "dogs"
]:

    print("Invalid prediction label")

    sys.exit(1)


# ============================================================
# Validate Probabilities
# ============================================================

if "probabilities" not in result:

    print("Probabilities field missing")

    sys.exit(1)


probabilities = result["probabilities"]


if "cats" not in probabilities:

    print("Cats probability missing")

    sys.exit(1)


if "dogs" not in probabilities:

    print("Dogs probability missing")

    sys.exit(1)


total_probability = (
    probabilities["cats"]
    + probabilities["dogs"]
)


if abs(total_probability - 1.0) > 0.01:

    print(
        "Invalid probabilities:",
        total_probability
    )

    sys.exit(1)


# ============================================================
# Final Result
# ============================================================

print()
print("======================================")
print("ALL SMOKE TESTS PASSED")
print("======================================")

sys.exit(0)