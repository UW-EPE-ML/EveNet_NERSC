from typing import Callable

import numpy as np
import torch
from evenet.utilities.diffusion_sampler import DDIMSampler
from functools import partial

import matplotlib.pyplot as plt
from evenet.network.loss.generation import loss as gen_loss
from evenet.utilities.debug_tool import time_decorator
from typing import Dict
import wandb

class NeutrinoGenerationMetrics:
    def __init__(self, device, class_names, feature_names, hist_xmin = -15, hist_xmax = 15, num_bins=60):
        self.sampler = DDIMSampler(device)
        self.device = device

        # Default values for histogram
        self.num_bins = num_bins
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax

        self.feature_names = feature_names

        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.num_classes = len(class_names)
        self.class_names = class_names

        self.histogram = dict()
        self.truth_histogram = dict()

    def update(self,
               model,
               input_set,
               num_steps=40,
               eta=1.0):

        model.eval()

        predict_distribution = dict()
        truth_distribution = dict()

        #####################################
        ## Generate invisible point cloud  ##
        #####################################

        data_shape = input_set['x_invisible'].shape
        process_id = input_set['classification']

        predict_for_neutrino = partial(
            model.predict_diffusion_vector,
            mode = "neutrino",
            cond_x=input_set,
            noise_mask = input_set["x_invisible_mask"].unsqueeze(-1) # [B, T, 1] to match noise x
        )


        generated_distribution = self.sampler.sample(
            data_shape=data_shape,
            pred_fn=predict_for_neutrino,
            normalize_fn=model.sequential_normalizer,
            eta=eta,
            num_steps=num_steps,
        )

        masking = input_set["x_invisible_mask"]
        for i in range(data_shape[-1]):
            predict_distribution[f"neutrino-{self.feature_names[i]}"] = generated_distribution[..., i]
            truth_distribution[f"neutrino-{self.feature_names[i]}"] = input_set['x_invisible'][..., i]

        #--------------- working line -----------------
        for distribution_name, distribution in predict_distribution.items():
            if distribution_name not in self.histogram:
                self.histogram[distribution_name] = {
                    class_name: np.zeros(self.num_bins)
                    for class_name in self.class_names
                }
            if distribution_name not in self.truth_histogram:
                self.truth_histogram[distribution_name]  = {
                    class_name: np.zeros(self.num_bins)
                    for class_name in self.class_names
                }

            for class_index, class_name in enumerate(self.class_names):
                class_mask = (process_id == class_index)
                if predict_distribution[distribution_name].size() == masking.size():
                    # Masking for point cloud
                    total_mask = masking[class_mask].flatten()
                    pred = predict_distribution[distribution_name][class_mask].flatten()[total_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten()[total_mask].detach().cpu().numpy()
                else:
                    pred = predict_distribution[distribution_name][class_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten().detach().cpu().numpy()
                hist, _ = np.histogram(pred, bins=self.bins)
                self.histogram[distribution_name][class_name] += hist

                hist, _ = np.histogram(truth, bins=self.bins)
                self.truth_histogram[distribution_name][class_name] += hist


    def reset(self):
        self.histogram = dict()
        self.truth_histogram = dict()

    def reduce_across_gpus(self):
        if torch.distributed.is_initialized():
            for name, hist_group in self.histogram.items():
                for class_name, hist in hist_group.items():
                    tensor = torch.tensor(hist, dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.histogram[name][class_name] = tensor.cpu().numpy()
            for name, hist_group in self.truth_histogram.items():
                for class_name, hist in hist_group.items():
                    tensor = torch.tensor(hist, dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.truth_histogram[name][class_name] = tensor.cpu().numpy()

    def plot_histogram_func(
            self,
            truth_histogram,
            histogram
    ):

        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        bin_widths = np.diff(self.bins)

        fig, ax = plt.subplots()

        for cls, cls_name in enumerate(self.class_names):
            # Plot training histogram (bars)
            counts = histogram[cls_name]
            if np.sum(counts) > 0:
                density = counts / (np.sum(counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Pred)"
                plt.plot(
                    self.bin_centers,
                    density,
                    color=color,
                    label=label,
                    linestyle='--',
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
            truth_counts = truth_histogram[cls_name]
            if np.sum(truth_counts) > 0:
                truth_density = truth_counts / (np.sum(truth_counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Truth)"
                plt.bar(
                    self.bin_centers,
                    truth_density,
                    width=bin_widths,
                    color=color,
                    alpha=0.7,
                    label=f"{cls_name} (Truth)", edgecolor=color, fill=False
                )

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()

        return fig

    def plot_histogram(self):
        figs = dict()
        for name in self.histogram:
            figs[name] = self.plot_histogram_func(
                self.truth_histogram[name],
                self.histogram[name]
            )
        return figs

@time_decorator(name="[Generation] shared_step")
def shared_step(
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        gen_metrics: GenerationMetrics,
        model: torch.nn.Module,
        loss_scale: float,
        device: torch.device,
        num_steps_global=20,
        num_steps_point_cloud=100,
        diffusion_on: bool = False,
):
    generation_loss = dict()
    total_gen_losses = 0
    for generation_target, generation_result in outputs.items():
        masking = batch["x_mask"].unsqueeze(-1) if generation_target == "point_cloud" else None
        generation_loss[generation_target] = gen_loss(
            predict=generation_result["vector"],
            target=generation_result["truth"],
            mask=masking
        )
        total_gen_losses += generation_loss[generation_target]

        if diffusion_on:
            gen_metrics.update(
                model=model,
                input_set=batch,
                num_steps_global=num_steps_global,
                num_steps_point_cloud=num_steps_point_cloud,
            )

    loss = total_gen_losses / len(outputs) * loss_scale
    return loss, generation_loss

@time_decorator(name="[Generation] shared_epoch_end")
def shared_epoch_end(
        global_rank,
        metrics_valid: GenerationMetrics,
        metrics_train: GenerationMetrics,
        logger,
):
    metrics_valid.reduce_across_gpus()
    if metrics_train:
        metrics_train.reduce_across_gpus()

    if global_rank == 0:
        figs = metrics_valid.plot_histogram()
        for name, fig in figs.items():
            logger.log({f"generation/{name}": wandb.Image(fig)})
            plt.close(fig)

    metrics_valid.reset()
    if metrics_train:
        metrics_train.reset()



