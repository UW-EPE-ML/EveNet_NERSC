import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List


class FAMO(nn.Module):
    def __init__(
        self,
        task_list: List[str],
        device: torch.device,
        lr: float = 0.025,
        gamma: float = 0.01,
    ):
        super().__init__()
        self.device = device
        self.task_list = task_list

        # Parameters (logits) for each task
        self.w = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([0.0], device=device))
            for k in task_list
        })

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=gamma)
        self.min_losses = {k: torch.tensor([0.0], device=device) for k in task_list}

        # Cache for update step
        self.prev_task_list = []
        self.prev_losses = None

    def step(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.prev_task_list = list(loss_dict.keys())
        losses = torch.stack([loss_dict[k].flatten()[0] for k in self.prev_task_list])  # [N]

        logits = torch.cat([self.w[k] for k in self.prev_task_list])  # [N]
        weights = F.softmax(logits, dim=0)  # [N]

        min_vec = torch.stack([self.min_losses[k].flatten()[0] for k in self.prev_task_list])
        D = losses - min_vec + 1e-8
        c = (weights / D).sum().detach()

        self.prev_losses = losses.detach()

        weighted_loss = (D.log() * weights / c).sum()
        return weighted_loss

    def update(self, current_loss_dict: Dict[str, torch.Tensor]) -> None:
        delta = []
        for k in self.prev_task_list:
            prev = self.prev_losses[self.prev_task_list.index(k)]
            curr = current_loss_dict[k].detach().flatten()[0]
            delta.append(
                torch.log(prev - self.min_losses[k].flatten()[0] + 1e-8)
                - torch.log(curr - self.min_losses[k].flatten()[0] + 1e-8)
            )

            print(f"prev: {prev} curr: {curr} delta: {delta[-1]}")
        delta = torch.stack(delta)  # [N]

        # Instead of using weights saved from step()
        with torch.enable_grad():
            logits = torch.stack([self.w[k] for k in self.prev_task_list]).squeeze(-1)
            weights = F.softmax(logits, dim=0)

            print(f"logits: {logits} weights: {weights}")

            grads = torch.autograd.grad(weights, [logits], grad_outputs=delta, retain_graph=False)[0]

        print(delta, weights, grads)

        self.optimizer.zero_grad()
        for i, k in enumerate(self.prev_task_list):
            self.w[k].grad = grads[i].unsqueeze(0)  # [1]

        self.optimizer.step()

    def log(self, prefix: str = "famo") -> Dict[str, float]:
        logits = torch.cat([self.w[k] for k in self.w])
        weights = F.softmax(logits, dim=0)
        log_data = {
            f"{prefix}/weight_entropy": -(weights * weights.log()).sum().item()
        }
        for i, k in enumerate(self.w):
            log_data[f"{prefix}/weight/{k}"] = weights[i].item()
            log_data[f"{prefix}/logit/{k}"] = self.w[k].item()
        return log_data
