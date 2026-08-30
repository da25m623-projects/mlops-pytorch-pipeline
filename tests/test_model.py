import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    inputs = torch.randn(2, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (2, 10)
