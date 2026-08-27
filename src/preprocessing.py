from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224


def preprocess_image(image):
    """
    Preprocess an input PIL image for the CNN.

    Returns a tensor of shape:
    (3, 224, 224)
    """

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image)