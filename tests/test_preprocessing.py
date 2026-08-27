import torch
from PIL import Image

from src.preprocessing import preprocess_image


def test_preprocess_image_shape():

    # Create a dummy RGB image
    image = Image.new(
        "RGB",
        (500, 300),
        color="white"
    )

    tensor = preprocess_image(image)

    assert isinstance(
        tensor,
        torch.Tensor
    )

    assert tensor.shape == (
        3,
        224,
        224
    )