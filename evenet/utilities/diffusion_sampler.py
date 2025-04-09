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

    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time_expanded)  # (B, 1, ...)
    perturbed_x = x * alpha + eps * sigma
    score = eps * alpha - x * sigma

    return perturbed_x, score


class DDIMSampler():
    def __init__(self):
        pass

    def prior_sde(self, dimensions) -> Tensor:
        return torch.randn(dimensions, dtype=torch.float32)

    def sample(
        self,
        input_set,
        data_shape,
        device,
        model,
        num_steps: int,
        eta: float = 1.0,
    ) -> Tensor:
        """
        time: time tensor (B,)
        """
        batch_size = data_shape[0]
        const_shape = (batch_size, *([1] * (len(data_shape) - 1)))
        x = self.prior_sde(data_shape)
        model.eval()
        for time_step in range(num_steps, 0, -1):
            t = torch.ones((batch_size, 1)).to(device) * time_step / num_steps
            t = t.float()  # Convert to float if needed
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shape=const_shape)

            t_prev = torch.ones((batch_size, 1), device=device) * (time_step - 1) / num_steps
            t_prev = t_prev.float()
            logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(t_prev, shape=const_shape)

            with torch.no_grad():
                input_set[""] = x
                # Compute the predicted epsilon using the model
                v = model(input_set, t.squeeze(-1)) # TODO
                eps = v * alpha + x * sigma

            # Update x using DDIM deterministic update rule
            pred_x0 = (x - sigma * eps) / alpha
            x = alpha_ * pred_x0 + sigma_ * (eta * eps)
        return x