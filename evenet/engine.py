import math
from functools import partial
from typing import Any, Dict, Union

import wandb
import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities.model_summary import summarize
from lion_pytorch import Lion
from matplotlib import pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from evenet.dataset.types import FeatureInfo
from evenet.network.evenet_model import EveNetModel

from evenet.network.metrics.general_comparison import GenericMetrics
from evenet.network.metrics.classification import ClassificationMetrics
from evenet.network.metrics.classification import shared_step as cls_step, shared_epoch_end as cls_end
from evenet.network.metrics.assignment import get_assignment_necessaries as get_ass
from evenet.network.metrics.assignment import shared_step as ass_step, shared_epoch_end as ass_end
from evenet.network.metrics.assignment import SingleProcessAssignmentMetrics
from evenet.network.metrics.generation import GenerationMetrics
from evenet.network.metrics.generation import shared_step as gen_step, shared_epoch_end as gen_end
from evenet.network.loss.grad_norm import GradNormController

from evenet.utilities.debug_tool import time_decorator, log_function_stats


def get_total_gradient(module, norm_type="l1"):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            if norm_type == "l1":
                total += param.grad.abs().sum().item()
            elif norm_type == "l2":
                total += (param.grad ** 2).sum().item()
    return total


class EveNetEngine(L.LightningModule):
    def __init__(self, global_config, world_size=1, total_events=1024):
        super().__init__()
        self.optimizer_name_map = None
        self.optimizer_name_list = []
        self.classification_metrics_train = None
        self.classification_metrics_valid = None
        self.assignment_metrics_train = None
        self.assignment_metrics_valid = None
        self.generation_metrics_train = None
        self.generation_metrics_valid = None
        self.model_parts = {}
        self.model: Union[EveNetModel, None] = None
        self.config = global_config
        self.world_size = world_size
        self.total_events = total_events
        self.pretrain_ckpt_path: str = global_config.options.Training.pretrain_model_load_path

        self.num_classes: list[str] = global_config.event_info.class_label.get('EVENT', {}).get('signal', [0])[0]

        ###### Initialize Keys for Data Inputs #####
        self.input_keys = ["x", "x_mask", "conditions", "conditions_mask"]
        self.target_classification_key = 'classification'
        self.target_regression_key = 'regression-data'
        self.target_regression_mask_key = 'regression-mask'
        self.target_assignment_key = 'assignments-indices'
        self.target_assignment_mask_key = 'assignments-mask'

        ###### Initialize Model Components Configs #####
        self.component_cfg = global_config.options.Training.Components

        self.classification_cfg = self.component_cfg.Classification
        self.regression_cfg = self.component_cfg.Regression
        self.assignment_cfg = self.component_cfg.Assignment
        self.generation_cfg = self.component_cfg.GlobalGeneration

        ###### Initialize Normalizations and Balance #####
        self.normalization_dict: dict = torch.load(self.config.options.Dataset.normalization_file)
        self.class_weight = self.normalization_dict["class_balance"]
        self.assignment_weight = self.normalization_dict["particle_balance"]
        self.subprocess_balance = self.normalization_dict["subprocess_balance"]

        print(f"{self.__class__.__name__} normalization dicts initialized")

        ###### Initialize Assignment Necessaries ######
        self.ass_args = None
        if self.assignment_cfg.include:
            print("configure permutation indices")
            self.permutation_indices = dict()
            self.num_targets = dict()

            self.ass_args = get_ass(global_config.event_info)

        #######  Initialize Diffusion Settings ##########
        self.diffusion_every_n_epochs = global_config.options.Training.diffusion_every_n_epochs
        self.diffusion_every_n_steps = global_config.options.Training.diffusion_every_n_steps

        self.global_diffusion_steps = self.component_cfg.GlobalGeneration.diffusion_steps
        self.point_cloud_diffusion_steps = self.component_cfg.EventGeneration.diffusion_steps

        ###### Initialize Loss ######
        self.grad_norm: Union[GradNormController, None] = None

        self.cls_loss = None
        if self.classification_cfg.include:
            import evenet.network.loss.classification as cls_loss
            self.cls_loss = cls_loss.loss
            print(f"{self.__class__.__name__} classification loss initialized")

        self.reg_loss = None
        if self.regression_cfg.include:
            import evenet.network.loss.regression as reg_loss
            self.reg_loss = reg_loss.loss
            print(f"{self.__class__.__name__} regression loss initialized")

        self.ass_loss = None
        if self.assignment_cfg.include:
            import evenet.network.loss.assignment as ass_loss
            assignment_loss_partial = partial(
                ass_loss.loss,
                focal_gamma=0.1,
                particle_balance=self.assignment_weight,
                process_balance=self.subprocess_balance,
                **self.ass_args['loss']
            )
            self.ass_loss = assignment_loss_partial
            print(f"{self.__class__.__name__} assignment loss initialized")

        self.gen_loss = None
        if self.generation_cfg.include:
            import evenet.network.loss.generation as gen_loss
            self.gen_loss = gen_loss.loss
            print(f"{self.__class__.__name__} generation loss initialized")

        ###### Initialize Optimizers ######
        self.hyper_par_cfg = {
            'batch_size': global_config.platform.batch_size,
            'epoch': global_config.options.Training.epochs,
            'lr_factor': global_config.options.Training.learning_rate_factor,
            'warm_up_factor': global_config.options.Training.learning_rate_warm_up_factor,
            'weight_decay': global_config.options.Training.weight_decay,
            'decoupled_weight_decay': global_config.options.Training.decoupled_weight_decay,
        }
        self.automatic_optimization = False

        ###### For general log ######
        self.general_log = GenericMetrics()

        ###### Progressive Training ######
        self.progressive_training: list = global_config.options.Training.get("ProgressiveTraining", [])
        self.progressive_index = 0

        ###### Last ######
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        pass

    @time_decorator()
    def shared_step(self, batch: Any, batch_idx: int, active_components: list[str]):
        batch_size = batch["x"].shape[0]
        device = self.device

        inputs = {
            key: value.to(device=device) for key, value in batch.items()
        }

        outputs = self.model.shared_step(
            batch=inputs,
            batch_size=batch_size,
        )

        loss = torch.zeros(batch_size, device=self.device, requires_grad=True)
        loss_head_dict = {}
        loss_detailed_dict = {}
        if self.classification_cfg.include:
            scaled_cls_loss = cls_step(
                target_classification=batch[self.target_classification_key].to(device=device),
                cls_output=next(iter(outputs["classification"].values())),
                cls_loss_fn=self.cls_loss,
                class_weight=self.class_weight.to(device=device),
                loss_dict=loss_head_dict,
                loss_scale=self.classification_cfg.loss_scale,
                metrics=self.classification_metrics_train if self.training else self.classification_metrics_valid,
                device=device,
            )

            if "classification" in active_components or active_components == []:
                loss = loss + scaled_cls_loss

        if self.regression_cfg.include:
            target_regression = batch[self.target_regression_key].to(device=device)
            target_regression_mask = batch[self.target_regression_mask_key].to(device=device)
            reg_output = outputs["regression"]
            reg_output = torch.cat([v.view(batch_size, -1) for v in reg_output.values()], dim=-1)
            reg_loss = self.reg_loss(
                reg_output,
                target_regression.float(),
                target_regression_mask.float(),
            )
            loss = loss + reg_loss * self.regression_cfg.loss_scale

            loss_head_dict["regression"] = reg_loss

        ass_predicts = None
        if self.assignment_cfg.include:
            ass_targets = batch[self.target_assignment_key].to(device=device)
            ass_targets_mask = batch[self.target_assignment_mask_key].to(device=device)
            scaled_ass_loss, ass_predicts = ass_step(
                ass_loss_fn=self.ass_loss,
                loss_dict=loss_head_dict,
                loss_detailed_dict=loss_detailed_dict,
                assignment_loss_scale=self.assignment_cfg.assignment_loss_scale,
                detection_loss_scale=self.assignment_cfg.detection_loss_scale,
                process_names=self.config.event_info.process_names,
                assignments=outputs["assignments"],
                detections=outputs["detections"],
                targets=ass_targets,
                targets_mask=ass_targets_mask,
                batch_size=batch_size,
                device=device,
                metrics=self.assignment_metrics_train if self.training else self.assignment_metrics_valid,
                point_cloud=inputs['x'],
                point_cloud_mask=inputs['x_mask'],
                subprocess_id=inputs["subprocess_id"],
                **self.ass_args['step']
            )
            loss_head_dict['assignment'] = scaled_ass_loss

            if "assignment" in active_components or active_components == []:
                loss = loss + scaled_ass_loss

        if self.generation_cfg.include:
            scaled_gen_loss, detailed_gen_loss = gen_step(
                batch=batch,
                outputs=outputs["generations"],
                gen_metrics=self.generation_metrics_train if self.training else self.generation_metrics_valid,
                model=self.model,
                loss_scale=self.generation_cfg.loss_scale,
                device=device,
                num_steps_global=self.global_diffusion_steps,
                num_steps_point_cloud=self.point_cloud_diffusion_steps,
                diffusion_on=(
                        not self.training
                        and ((self.current_epoch % self.diffusion_every_n_epochs) == (
                        self.diffusion_every_n_epochs - 1))
                        and ((batch_idx % self.diffusion_every_n_steps) == 0)
                )
            )
            loss_head_dict["generation"] = scaled_gen_loss

            if "generation" in active_components or active_components == []:
                loss = loss + scaled_gen_loss

        self.general_log.update(loss_detailed_dict, is_train=self.training)

        return loss, loss_head_dict, loss_detailed_dict, ass_predicts

    @time_decorator()
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        step = self.global_step
        epoch = self.current_epoch

        # Get all optimizers and schedulers
        optimizers = list(self.optimizers())
        schedulers = self.lr_schedulers()
        active_components = []

        if len(self.progressive_training) > 0:
            active_progress = self.progressive_training[0]

            optimizers = [
                optimizers[idx]
                for idx, name in enumerate(self.model_parts.keys())
                if name in active_progress['components']
            ]
            schedulers = [
                schedulers[idx]
                for idx, name in enumerate(self.model_parts.keys())
                if name in active_progress['components']
            ]
            active_components = active_progress['components']

        loss, loss_head, loss_dict, _ = self.shared_step(
            batch=batch, batch_idx=batch_idx, active_components=active_components
        )

        self.log("train/loss", loss.mean(), prog_bar=True, sync_dist=True)
        for name, val in loss_head.items():
            self.log(f"train/{name}", val.mean(), prog_bar=False, sync_dist=True)

        for name, val in loss_dict.items():
            for n, v in val.items():
                self.log(f"{n}/train/{name}", v.mean(), prog_bar=False, sync_dist=True)

        # === Manual optimization ===
        for opt in optimizers:
            opt.zero_grad()

        self.safe_manual_backward(loss.mean())

        self.check_gradient()

        for opt in optimizers:
            opt.step()

        for sch in schedulers:
            sch.step()

        return loss.mean()

    # @time_decorator
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # Implement your validation step logic here
        step = self.global_step
        epoch = self.current_epoch
        # print(f"[Epoch {epoch} | Step {step}] {self.__class__.__name__} validation step")

        active_components = []
        if len(self.progressive_training) > 0:
            active_progress = self.progressive_training[0]
            active_components = active_progress['components']

        loss, loss_head, loss_dict, _ = self.shared_step(
            batch=batch, batch_idx=batch_idx, active_components=active_components
        )

        self.log("val/loss", loss.mean(), prog_bar=True, sync_dist=True)

        for name, val in loss_head.items():
            self.log(f"val/{name}", val.mean(), prog_bar=False, sync_dist=True)

        for name, val in loss_dict.items():
            for n, v in val.items():
                self.log(f"{n}/val/{name}", v.mean(), prog_bar=False, sync_dist=True)

        return loss.mean()

    def check_gradient(self):
        # Define the heads you want to track
        gradient_heads = {}

        if self.classification_cfg.include:
            gradient_heads["classification"] = self.model.Classification

        if self.regression_cfg.include:
            gradient_heads["regression"] = self.model.Regression

        if self.generation_cfg.include:
            gradient_heads["generation-global"] = self.model.GlobalGeneration
            gradient_heads["generation-event"] = self.model.EventGeneration

        if self.assignment_cfg.include:
            assignment_heads = self.model.Assignment.multiprocess_assign_head
            gradient_heads.update({
                f'assignment_{name}': assignment_heads[name]
                for name in assignment_heads.keys()
            })

        for name, module in gradient_heads.items():
            grad_mag = get_total_gradient(module)
            num_params = sum(p.numel() for p in module.parameters() if p.grad is not None)
            grad_avg = grad_mag / num_params if num_params > 0 else 0.0
            self.log(f"grad_head/{name}", grad_avg, prog_bar=False, sync_dist=True)

    # @time_decorator
    def on_fit_start(self) -> None:

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                print(f"[Rank {torch.distributed.get_rank()}] {name}: {param.view(-1)[:5]}")

            if i == 3:
                break

        if self.classification_cfg.include:
            self.classification_metrics_train = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )
            self.classification_metrics_valid = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )

        if self.assignment_cfg.include:
            def make_assignment_metrics():
                return {
                    process: SingleProcessAssignmentMetrics(
                        device=self.device,
                        event_permutations=self.config.event_info.event_permutations[process],
                        event_symbolic_group=self.config.event_info.event_symbolic_group[process],
                        event_particles=self.config.event_info.event_particles[process],
                        product_symbolic_groups=self.config.event_info.product_symbolic_groups[process],
                        ptetaphimass_index=self.config.event_info.ptetaphimass_index,
                        process=process
                    ) for process in self.config.event_info.process_names
                }

            self.assignment_metrics_train = make_assignment_metrics()
            self.assignment_metrics_valid = make_assignment_metrics()

        if self.generation_cfg.include:
            self.generation_metrics_train = GenerationMetrics(
                class_names=self.config.event_info.class_label['EVENT']['signal'][0],
                feature_names=self.config.event_info.sequential_feature_names,
                device=self.device
            )
            self.generation_metrics_valid = GenerationMetrics(
                class_names=self.config.event_info.class_label['EVENT']['signal'][0],
                feature_names=self.config.event_info.sequential_feature_names,
                device=self.device
            )

    def on_fit_end(self) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.general_log.reduce_across_gpus(
            device=device
        )

        if self.global_rank == 0:
            print(f"{self.__class__.__name__} on_fit_end")

            figs = self.general_log.plot_all()

            for metric, fig in figs.items():
                self.logger.experiment.log({
                    f"General/{metric}": wandb.Image(fig)
                })
                plt.close(fig)

            # debug time information
            log_function_stats(self.logger)

            summary = summarize(self, max_depth=3)
            columns = ["Name", "Type", "Params"]
            data = [
                [str(name), str(type_), int(num)]
                for name, type_, num in zip(summary.layer_names, summary.layer_types, summary.param_nums)
            ]
            self.logger.log_table(
                key="model summary",
                columns=columns,
                data=data,
            )

            wandb.finish()  # ensure everything is flushed
            print("W&B logging done and finished.")

        print(f"{self.__class__.__name__} on_fit_end all end [Rank: {self.global_rank}]")

    def on_validation_start(self):
        pass

    def on_train_epoch_start(self) -> None:
        pass

    @time_decorator()
    def on_validation_epoch_end(self) -> None:
        if self.classification_cfg.include:
            cls_end(
                global_rank=self.global_rank,
                metrics_valid=self.classification_metrics_valid,
                metrics_train=self.classification_metrics_train,
                num_classes=self.num_classes,
                logger=self.logger.experiment,
            )

        if self.assignment_cfg.include:
            ass_end(
                global_rank=self.global_rank,
                metrics_valid=self.assignment_metrics_valid,
                metrics_train=self.assignment_metrics_train,
                logger=self.logger.experiment,
            )

        if self.generation_cfg.include:
            gen_end(
                global_rank=self.global_rank,
                metrics_valid=self.generation_metrics_valid,
                metrics_train=self.generation_metrics_train,
                logger=self.logger.experiment
            )

        self.general_log.finalize_epoch(is_train=False)

    @time_decorator()
    def on_train_epoch_end(self) -> None:
        self.general_log.finalize_epoch(is_train=True)

        self.log("progress", self.progressive_index, prog_bar=True, sync_dist=False)

        if len(self.progressive_training) > 0:
            print(f"--> Current Epoch: {self.current_epoch}, Target: {self.progressive_training[0]['epoch']}")
            if self.current_epoch + 1 == self.progressive_training[0]['epoch']:
                self.progressive_training.pop(0)
                self.progressive_index += 1

        pass

    @time_decorator()
    def safe_manual_backward(self, loss, *args, **kwargs):
        super().manual_backward(loss, *args, **kwargs)
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"🚨 Gradient in {name} is NaN or Inf!")
                    raise ValueError("Gradient check failed.")

    # @time_decorator
    def configure_optimizers(self):
        cfg = self.hyper_par_cfg
        lr_factor = cfg.get('lr_factor', 1.0)
        batch_size = cfg['batch_size']
        epochs = cfg['epoch']
        warm_up_factor = cfg.get('warm_up_factor', 0.5)

        # Distributed training info
        world_size = self.world_size
        dataset_size = self.total_events
        steps_per_epoch = math.ceil(dataset_size / (batch_size * world_size))
        warmup_steps = warm_up_factor * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        print(f"--> Optimizer Configuration:")
        print("word_size: ", world_size)
        print("dataset_size: ", dataset_size)
        print("steps_per_epoch: ", steps_per_epoch)
        print("warmup_steps: ", warmup_steps)
        print("total_steps: ", total_steps)

        betas = (0.95, 0.99)

        def create_optim_schedule(p, base_lr, warm_up: bool = True):
            scaled_lr = base_lr * math.sqrt(world_size) / lr_factor
            optimizer = Lion(
                p,
                lr=scaled_lr,
                betas=betas
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps if warm_up else 0,
                num_training_steps=total_steps,
                num_cycles=0.5
            )
            return optimizer, scheduler

        optimizers, schedulers = [], []
        for name, modules in self.model_parts.items():
            print(f"Configuring optimizer/scheduler for {name} with modules: {modules['modules']}")

            if not modules:
                continue
            valid_modules = [getattr(self.model, m) for m in modules['modules'] if m is not None]
            params = [p for m in valid_modules for p in m.parameters()]

            if len(params) == 0:
                print(f"Warning: No parameters found for {name}. Skipping optimizer/scheduler configuration.")
                continue

            opt, sch = create_optim_schedule(params, base_lr=modules['lr'], warm_up=modules['warm_up'])
            optimizers.append(opt)
            schedulers.append({
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": f"lr-{name}"
            })
            self.optimizer_name_list.append(name)

        self.optimizer_name_map = {
            name: idx for idx, name in enumerate(self.optimizer_name_list)
        }

        return optimizers, schedulers

    # @time_decorator
    def configure_model(self) -> None:
        print(f"{self.__class__.__name__} configure model on device {self.device}")
        if self.model is not None:
            return

        self.model = EveNetModel(
            config=self.config,
            device=self.device,
            classification=self.classification_cfg.include,
            regression=self.regression_cfg.include,
            generation=self.generation_cfg.include,
            assignment=self.assignment_cfg.include,
            normalization_dict=self.normalization_dict,
        )

        if self.pretrain_ckpt_path is not None:
            if self.global_rank == 0:
                print(f"Loading PRETRAIN model from: {self.pretrain_ckpt_path}")
                state_dict = torch.load(self.pretrain_ckpt_path, map_location=self.device)['state_dict']
                # Remove "model." prefix from keys
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print("--> Missing keys:", missing)
                print("--> Unexpected keys:", unexpected)

        # self.logger.experiment.watch(self.model, log="all", log_graph=True, log_freq=500)
        for heads in [
            # self.model.Assignment.multiprocess_assign_head,
            # self.model.Assignment.multiprocess_assign_head.SingleVisibleDecay,
            # self.model.Assignment.multiprocess_assign_head.DiObjectDecay,
            # self.model.Assignment.multiprocess_assign_head.LeptonicTop,
            # self.model.Assignment.multiprocess_assign_head.HadronicTop,
        ]:
            self.logger.experiment.watch(heads, log="gradients", log_freq=10, log_graph=False)

        # Define Freezing
        # self.model.freeze_module("Classification", self.classification_cfg.get("freeze", {}))
        # self.model.freeze_module("Regression", self.regression_cfg.get("freeze", {}))
        # self.model.freeze_module("Assignment", self.assignment_cfg.get("freeze", {}))

        # Define model part groups
        self.model_parts = {}

        for key, cfg in self.component_cfg.items():
            group = cfg.get("optimizer_group", None)
            if not group:
                continue

            module_attr = getattr(self.model, key, None)
            if not module_attr:
                print(f"⚠️ Warning: No module set for '{key}'. Skipping.")
                continue

            # Add to group
            if group not in self.model_parts:
                self.model_parts[group] = {
                    'lr': cfg['learning_rate'],
                    'warm_up': cfg.get('warm_up', True),
                    'modules': [],
                }
            self.model_parts[group]['modules'].append(key)

        print("model parts: ", self.model_parts)

        ### Progressive Training ###
        print("--> Progressive Training:")
        sum_epochs = 0
        for i, progress in enumerate(self.progressive_training):
            ratio = int(self.progressive_training[i]['epoch_ratio'] * self.hyper_par_cfg['epoch'])
            self.progressive_training[i]['epoch'] = ratio + sum_epochs
            sum_epochs += ratio

            print(f"{progress['name']}: {ratio: .0f} epochs, total {sum_epochs: .0f} epochs")
            for component in progress['components']:
                optimizer_exists = self.model_parts.get(component, None)
                if optimizer_exists:
                    print(f"  --> Optimizer: {component} ✅")
                else:
                    print(f"  --> Optimizer: {component} ❌")

        ### Initialize Gradient Norm ###
        if self.grad_norm is None and self.config.options.Training.get("GradientNorm", False):
            self.grad_norm = GradNormController(
                task_names=[
                    "classification",
                    "regression",
                    *[name for name in self.config.resonance],
                ],
                **self.config.options.Training.GradientNorm,
            )
            self.register_module("grad_norm", self.grad_norm)

            print(f"{self.__class__.__name__} GRADIENT NORM Applied!")
            print(f"  --> GRADIENT NORM List: {self.grad_norm.task_names}")

    def on_save_checkpoint(self, checkpoint):
        orig_model = getattr(self.model, "_orig_mod", self.model)
        new_sd = {f"model.{k}": v for k, v in orig_model.state_dict().items()}
        checkpoint["state_dict"] = new_sd

        pass
