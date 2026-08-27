import torch
import torch.nn as nn

from src.model_utils import predict_class


class DummyModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(
            3 * 224 * 224,
            2
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        return self.linear(x)


def test_predict_class():

    model = DummyModel()

    image_tensor = torch.randn(
        1,
        3,
        224,
        224
    )

    label, probabilities = predict_class(
        model,
        image_tensor
    )

    assert label in [
        "cats",
        "dogs"
    ]

    assert "cats" in probabilities
    assert "dogs" in probabilities

    assert len(probabilities) == 2

    total_probability = sum(
        probabilities.values()
    )

    assert abs(
        total_probability - 1.0
    ) < 1e-5