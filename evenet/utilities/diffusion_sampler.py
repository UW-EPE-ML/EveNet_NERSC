import torch
from torch import Tensor

def logsnr_schedule_cosine(time: Tensor,
                           logsnr_min: float = -20.,
                           logsnr_max: float = 20.) -> Tensor:
    logsnr_min = Tensor([logsnr_min]).to(time.device)
    logsnr_max = Tensor([logsnr_max]).to(time.device)
    b = torch.atan(torch.exp(-0.5 * logsnr_max)).to(time.device)
    a = (torch.atan(torch.exp(-0.5 * logsnr_min)) - b).to(time.device)
    return -2.0 * torch.log(torch.tan(a * time.to(torch.float32) + b))

def get_logsnr_alpha_sigma(time: Tensor,
                           shape=None):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

    if shape is not None:
        logsnr = logsnr.view(shape).to(torch.float32)
        alpha = alpha.view(shape).to(torch.float32)
        sigma = sigma.view(shape).to(torch.float32)

    return logsnr, alpha, sigma

def add_noise(x: Tensor,
              time: Tensor) -> Tensor:
    """
    x: input tensor,
    time: time tensor (B,)
    """
    eps = torch.randn_like(x)
    time_expanded = time.view(time.shape[0], *([1] * (x.dim() - 1)))

    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time_expanded) # (B, 1, ...)
    perturbed_x = x * alpha + eps * sigma
    score = eps * alpha - x * sigma

    return perturbed_x, score