import torch
import torch.nn.functional as F

# ==========================================================
# BASE CAM
# ==========================================================

class BaseCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _normalize(self, cam):
        cam = F.relu(cam)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# ==========================================================
# GradCAM++
# ==========================================================

class GradCAMPlusPlus(BaseCAM):

    def generate(self, image, metadata, target_class):

        image.requires_grad_(True)

        output = self.model(image, metadata)
        score = output[:, target_class]

        grads = torch.autograd.grad(
            score,
            self.activations,
            retain_graph=True,
            create_graph=True
        )[0]

        grads2 = grads ** 2
        grads3 = grads ** 3

        denominator = (
            2 * grads2 +
            torch.sum(self.activations * grads3,
                      dim=(2,3), keepdim=True) + 1e-8
        )

        alpha = grads2 / denominator
        weights = torch.sum(alpha * F.relu(grads),
                            dim=(2,3), keepdim=True)

        cam = torch.sum(weights * self.activations,
                        dim=1, keepdim=True)

        cam = self._normalize(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return cam.squeeze().detach().cpu().numpy()