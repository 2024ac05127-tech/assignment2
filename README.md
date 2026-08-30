[README.md](https://github.com/user-attachments/files/31617619/README.md)
# MLOps Assignment 2 – Cats vs Dogs Classification

## 1. Project Overview

This project implements an end-to-end **MLOps pipeline** for binary image classification of **cats and dogs** for a pet adoption platform.

The workflow covers:

- Dataset preparation and preprocessing
- CNN model development
- Experiment tracking and evaluation
- FastAPI inference service
- Docker containerization
- Automated testing
- CI/CD using GitHub Actions
- Docker image publishing
- Continuous deployment using Docker Compose
- Post-deployment smoke testing
- Monitoring, request logging, and basic performance metrics

---

## 2. Use Case

The objective is to automatically classify an uploaded pet image as either:

- **Cat**
- **Dog**

The inference service accepts an image and returns the predicted class together with the probability for each class.

---

## 3. Model Development

A baseline convolutional neural network (**BaselineCNN**) was implemented using PyTorch.

### Input Processing

Images are:

- Resized to **224 × 224**
- Converted to RGB
- Converted to tensors
- Normalized using ImageNet-style mean and standard deviation

### Model Output

The model performs binary classification with two output classes:

```text
cats
dogs
```

The trained model is saved as:

```text
models/baseline_cnn.pt
```

---

## 4. Model Performance

### Training and Validation Accuracy

The training accuracy increases from approximately **0.58 to 0.70**, while validation accuracy increases from approximately **0.63 to 0.74** over three epochs.

![Training and Validation Accuracy](accuracy_curve.png)

### Training and Validation Loss

Both training and validation loss decrease over the three epochs, indicating that the model is learning during training.

![Training and Validation Loss](loss_curve.png)

### Confusion Matrix

The confusion matrix shows the classification results for cats and dogs.

- Actual cats predicted as cats: **984**
- Actual cats predicted as dogs: **266**
- Actual dogs predicted as cats: **378**
- Actual dogs predicted as dogs: **872**

![Cats vs Dogs Confusion Matrix](confusion_matrix.png)

Based on these values, the model correctly classifies a substantial number of both cats and dogs, while the remaining errors represent opportunities for further model improvement.

---

## 5. Inference API

The model is exposed through a **FastAPI REST service**.

### Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "BaselineCNN",
  "model_file": "baseline_cnn.pt",
  "device": "cpu"
}
```

### Prediction Endpoint

```http
POST /predict
```

The endpoint accepts an image file and returns:

```json
{
  "filename": "test.jpg",
  "label": "cats",
  "probabilities": {
    "cats": 0.8102,
    "dogs": 0.1898
  }
}
```

---

## 6. Monitoring and Logging

Basic monitoring was added to the inference service.

The service tracks:

- Total request count
- Prediction count
- Error count
- Average prediction latency

The monitoring endpoint is:

```http
GET /metrics
```

Example:

```json
{
  "request_count": 0,
  "prediction_count": 0,
  "error_count": 0,
  "average_latency_seconds": 0.0
}
```

The application also records request and prediction information in the service logs without storing sensitive user information.

---

## 7. Docker Containerization

The inference service is packaged as a Docker image:

```text
krishnapr1/cats-dogs-api:latest
```

The service is exposed on port **8000**.

Example:

```bash
docker pull krishnapr1/cats-dogs-api:latest
```

The application can then be deployed using Docker Compose.

---

## 8. CI/CD Pipeline

GitHub Actions is used for automation.

### CI Pipeline

The CI workflow performs tasks such as:

1. Checkout source code
2. Install Python dependencies
3. Run automated tests
4. Build the Docker image
5. Publish the image to Docker Hub

### CD Pipeline

The CD workflow:

1. Runs on changes to the `main` branch
2. Pulls the latest Docker image
3. Deploys the application using Docker Compose
4. Checks the running container
5. Waits for application startup
6. Runs smoke tests

---

## 9. Smoke Testing

A smoke-test script verifies the deployed application.

The tests check:

1. `/health` endpoint
2. `/predict` endpoint

Successful execution produces:

```text
Testing health endpoint...
Health check PASSED

Testing prediction endpoint...
Prediction request PASSED

======================================
ALL SMOKE TESTS PASSED
======================================
```

---

## 10. Deployment Architecture

```text
Developer
    |
    | git push
    v
GitHub Repository
    |
    v
GitHub Actions
    |
    +----> Automated Tests
    |
    +----> Docker Build
    |
    +----> Docker Hub
              |
              v
       Docker Compose
              |
              v
       FastAPI Inference API
              |
        +-----+-----+
        |           |
     /health     /predict
        |
     /metrics
```

---

## 11. Project Structure

A typical project structure is:

```text
assignment2/
│
├── src/
│   └── inference.py
│
├── models/
│   └── baseline_cnn.pt
│
├── tests/
│   └── ...
│
├── smoke_tests.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
└── README.md
```

---

## 12. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Image Processing | Pillow / Torchvision |
| API | FastAPI |
| Server | Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| Deployment | Docker Compose |
| CI/CD | GitHub Actions |
| Container Registry | Docker Hub |
| Version Control | Git |
| Dataset Versioning | DVC / Git-LFS |
| Monitoring | Application logs and custom metrics |

---

## 13. Conclusion

The project demonstrates a complete MLOps workflow from model development to deployment and monitoring.

The trained CNN model is packaged as a REST API, containerized using Docker, automatically tested and published through CI, deployed through CD, and verified using post-deployment smoke tests. Basic request logging and application metrics provide visibility into the deployed inference service.

The workflow can be extended in future work with Prometheus/Grafana dashboards, model drift detection, larger training runs, and automated post-deployment model performance monitoring.
