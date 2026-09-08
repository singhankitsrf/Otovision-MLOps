import torch
from otovision.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 5)
