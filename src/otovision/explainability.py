from __future__ import annotations

import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_index

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    import matplotlib.cm as cm

    colored = cm.jet(heatmap)[..., :3]
    image = rgb_image.astype(float)
    if image.max() > 1.0:
        image = image / 255.0
    blended = (1 - alpha) * image + alpha * colored
    return np.clip(blended, 0.0, 1.0)
