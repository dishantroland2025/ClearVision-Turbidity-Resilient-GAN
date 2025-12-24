import torch
import torch.nn as nn
import kornia.color as K
import kornia.filters as KF


# -------------------------------------------------
# Adversarial Loss (LSGAN)
# -------------------------------------------------
def adversarial_loss(pred, target_is_real=True):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return torch.mean((pred - target) ** 2)


# -------------------------------------------------
# Pixel Loss (L1)
# -------------------------------------------------
def pixel_loss(fake, real):
    return torch.mean(torch.abs(fake - real))


# -------------------------------------------------
# LAB Color Loss (a, b channels only)
# -------------------------------------------------
def lab_color_loss(fake, real):
    fake_lab = K.rgb_to_lab((fake + 1) / 2)
    real_lab = K.rgb_to_lab((real + 1) / 2)

    return (
        torch.mean(torch.abs(fake_lab[:, 1] - real_lab[:, 1])) +
        torch.mean(torch.abs(fake_lab[:, 2] - real_lab[:, 2]))
    )


# -------------------------------------------------
# Edge Loss (Sobel)
# -------------------------------------------------
def edge_loss(fake, real):
    fake_edges = KF.sobel(fake)
    real_edges = KF.sobel(real)
    return torch.mean(torch.abs(fake_edges - real_edges))


# -------------------------------------------------
# Depth-Weighted Loss
# -------------------------------------------------
def depth_weighted_loss(fake, real, depth, max_depth=1.0):
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    weights = 1.0 + 4.0 * (depth / max_depth)
    return torch.mean(weights * torch.abs(fake - real))


# -------------------------------------------------
# Perceptual Loss (VGG-based, WITH normalization)
# -------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.layers = [2, 7, 16, 25]  # relu1_2, relu2_2, relu3_4, relu4_4

        for p in self.vgg.parameters():
            p.requires_grad = False

    def _normalize(self, x):
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=x.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=x.device
        ).view(1, 3, 1, 1)

        x = (x + 1) / 2  # [-1, 1] → [0, 1]
        return (x - mean) / std

    def forward(self, fake, real):
        fake = self._normalize(fake)
        real = self._normalize(real)

        loss = torch.tensor(0.0, device=fake.device)
        x_f, x_r = fake, real

        for i, layer in enumerate(self.vgg):
            x_f = layer(x_f)
            x_r = layer(x_r)
            if i in self.layers:
                loss += torch.mean(torch.abs(x_f - x_r))

        return loss


# -------------------------------------------------
# Generator Loss (Combined)
# -------------------------------------------------
def generator_loss(
    D,
    real_img,
    fake_img,
    input_img,
    depth=None,
    max_depth=1.0,
    perceptual_fn=None,
    lambdas=None
):
    # Adversarial
    pred_fake = D(input_img, fake_img)
    loss_adv = adversarial_loss(pred_fake, True)

    # Core losses
    loss_pix = pixel_loss(fake_img, real_img)
    loss_color = lab_color_loss(fake_img, real_img)
    loss_edge = edge_loss(fake_img, real_img)
    loss_perc = perceptual_fn(fake_img, real_img) if perceptual_fn is not None else 0

    # Depth (optional)
    loss_depth = (
        depth_weighted_loss(fake_img, real_img, depth, max_depth)
        if depth is not None else 0
    )

    # Weighted sum
    total = (
        lambdas["adv"] * loss_adv +
        lambdas["pixel"] * loss_pix +
        lambdas["color"] * loss_color +
        lambdas["edge"] * loss_edge +
        lambdas["perc"] * loss_perc +
        lambdas["depth"] * loss_depth
    )

    # -------------------------------------------------
    # Loss dictionary (ADDED AS REQUESTED)
    # -------------------------------------------------
    loss_dict = {
        "Total": total.item(),
        "Adv": loss_adv.item(),
        "Pixel": loss_pix.item(),
        "Color": loss_color.item(),
        "Edge": loss_edge.item(),
        "Depth": loss_depth.item() if depth is not None else 0
    }

    return total, loss_dict
