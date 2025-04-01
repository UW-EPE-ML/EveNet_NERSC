from typing import Dict
import torch
from torch import Tensor

from evenet.control.config import DotDict
from evenet.dataset.types import Batch
from evenet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork



class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, config: DotDict, torch_script: bool = False, **kwargs):
        super(JetReconstructionTraining, self).__init__(config, torch_script, **kwargs)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = {process: list(self.event_info.product_particles[process].keys()) for process in self.process_names}
        self.product_particle_names = {
            process: {
                particle: self.event_info.product_particles[process][particle][0]
                for particle in self.event_particle_names[process]
            }
            for process in self.event_particle_names
        }

        # Set to track used parameters
        self.used_params = set()

    def debug_unused_parameters(self):
        """Check for unused parameters after each forward pass."""
        print(f"Debugging unused parameters (batch level):")
        for name, param in self.named_parameters():
            if not param.requires_grad:  continue
            if name not in self.used_params:
                print(f"Unused parameter: {name}")

        # Reset used_params for the next batch
        self.used_params.clear()

    def on_fit_start(self):
        """Register hooks to track parameter usage."""
        for name, param in self.named_parameters():
            if param.requires_grad:  # Register hooks only if gradient is required

                def hook(*args, name=name):
                    self.used_params.add(name)
                param.register_hook(hook)

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:

        total_loss = []
        total_loss += self.loss_func(batch, batch_nb, "classifier", "train")
        if (self.options.Training.generation_loss_scale > 0 or self.options.Training.feature_generation_loss_scale > 0):
          total_loss += self.loss_func(batch, batch_nb, "generator", "train")
        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])
        self.log("train/loss/total_loss", total_loss.sum(), sync_dist=True)


        return total_loss.mean()

#    def on_after_backward(self):
#        self.debug_unused_parameters()
#        for name, param in self.named_parameters():
#            if not param.requires_grad: continue
#            if name not in self.used_params:
#                # Assign a dummy gradient (zeros or other values)
#                param.grad = torch.zeros_like(param)
