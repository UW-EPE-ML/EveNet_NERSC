import json
import math
import pickle

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import torch
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EveNetModel

from evenet.network.loss.classification import loss as cls_loss
from evenet.network.loss.regression import loss as reg_loss
from evenet.network.loss.assignment import loss as ass_loss
from evenet.network.loss.generation import loss as gen_loss
from evenet.network.loss.assignment import convert_target_assignment

from evenet.network.metrics.assignment import predict
from evenet.network.metrics.assignment import SingleProcessAssignmentMetrics

from preprocessing.preprocess import unflatten_dict

import torch
from evenet.network.metrics.classification import ClassificationMetrics
from matplotlib import pyplot as plt
from evenet.utilities.debug_tool import DebugHookManager
########################
## Debug configuration ##
########################

wandb_enable = True
n_epoch = 10
debugger_enable = False
device = "cuda"

workspacedir = "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output"

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/local_test.yaml")
num_classes = global_config.event_info.class_label['EVENT']['signal'][0]

shape_metadata = json.load(
    open(f"{workspacedir}/shape_metadata.json"))

normalization_dict = torch.load(f"{workspacedir}/normalization.pt")
particle_balance_dict = None # normalization_dict['particle_balance']

# Load the Parquet file locally
df = pq.read_table(
    f"{workspacedir}/data_run_yulei_11.parquet").to_pandas()
# Optional: Subsample for speed

df.sample(frac=1).reset_index(drop=True)
df_number = 512 # len(df) // 2
df = df.head(df_number)

# Assignment setting
# Record attention head basic information
event_info = global_config.event_info
permutation_indices = dict()
num_targets = dict()
event_permutation = dict()
event_particles = dict()
import re
from evenet.utilities.group_theory import complete_indices, symmetry_group

for process in global_config.event_info.process_names:
    permutation_indices[process] = []
    num_targets[process] = []
    for event_particle_name, product_symmetry in global_config.event_info.product_symmetries[process].items():
        topology_name = ''.join(global_config.event_info.product_particles[process][event_particle_name].names)
        topology_name = f"{event_particle_name}/{topology_name}"
        topology_name = re.sub(r'\d+', '', topology_name)
        topology_category_name = global_config.event_info.pairing_topology[topology_name]["pairing_topology_category"]
        permutation_indices_tmp = complete_indices(
            global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].degree,
            global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].permutations
        )
        permutation_indices[process].append(permutation_indices_tmp)
        event_particles[process] = [p for p in event_info.event_particles[process].names]
        event_permutation[process] = complete_indices(
            event_info.event_symmetries[process].degree,
            event_info.event_symmetries[process].permutations
        )
        permutation_group = symmetry_group(permutation_indices_tmp)
        num_targets[process].append(
            global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].degree)

# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(
    batch,
    shape_metadata=shape_metadata,
    unflatten=unflatten_dict
)


# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)
torch_batch = {
    k: v.to(device) for k, v in torch_batch.items()
}
# with open("/Users/avencastmini/PycharmProjects/EveNet/workspace/normalization_file/PreTrain_norm.pkl", 'rb') as f:
#     normalization_file = pickle.load(f)

num_splits = df_number // 128

# Run forward
model = EveNetModel(
    config=global_config,
    device=torch.device(device),
    normalization_dict=normalization_dict,
    assignment=True,
    generation=True
).to(device)

# model.freeze_module("Classification", global_config.options.Training.Components.Classification.get("freeze", {}))
# model.freeze_module("Regression", global_config.options.Training.Components.Regression.get("freeze", {}))

# model = MLP()

model.train()

debugger = DebugHookManager(track_forward=True, track_backward=True, save_values=True)
if debugger_enable:
    debugger.attach_hooks(model)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

batch_size = len(df)
split_batches = []

confusion_accumulator = ClassificationMetrics(
    num_classes=len(num_classes),
    normalize=True,
    device=torch.device(device),
)

for i in range(num_splits):
    start = math.floor(i * batch_size / num_splits)
    end = math.floor((i + 1) * batch_size / num_splits)
    split = {k: v[start:end] for k, v in torch_batch.items()}
    split_batches.append(split)

if wandb_enable:
    import wandb

    wandb.init(
        project="debug",
        entity=global_config.wandb.entity
    )

assignment_metrics = {process: SingleProcessAssignmentMetrics(
    device=device,
    event_permutations=event_info.event_permutations[process],
    event_symbolic_group=event_info.event_symbolic_group[process],
    event_particles=event_info.event_particles[process],
    product_symbolic_groups=event_info.product_symbolic_groups[process],
    ptetaphimass_index=event_info.ptetaphimass_index,
    process=process
) for process in event_info.process_names}

for iepoch in range(n_epoch):
    # if (n_epoch > 2): model.eval()
    for i, batch in enumerate(split_batches):
        with torch.autograd.set_detect_anomaly(True):
            # if True:
            for name, tensor in batch.items():
                if torch.isnan(tensor).any():
                    print(f"[Epoch {iepoch} / Batch {i}] NaN found in input tensor: {name}")

            outputs = model.shared_step(batch, batch_size=len(batch["classification"]))

            # outputs = model.shared_step(batch)

            # Regression
            # reg_output = outputs["regression"]
            # flattened = torch.cat([v.squeeze(0) for v in reg_output.values()], dim=-1)
            # if torch.isnan(flattened).any():
            #     print(f"[Epoch {iepoch} / Batch {i}] NaN in regression output")
            #
            # reg_target = batch["regression-data"].float()
            # reg_mask = batch["regression-mask"].float()
            #
            # r_loss = reg_loss(predict=flattened, target=reg_target, mask=reg_mask)
            # if torch.isnan(r_loss).any():
            #     print(f"[Epoch {iepoch} /Batch {i}] NaN in regression loss")

            # Classification
            cls_output = next(iter(outputs["classification"].values()))
            cls_target = batch["classification"]
            # cls_output = torch.nn.Softmax(dim=1)(cls_output)
            preds = cls_output.argmax(dim=-1)


            c_loss = cls_loss(predict=cls_output, target=cls_target, class_weight=normalization_dict['class_balance'].to(device))
            if torch.isnan(c_loss).any():
                print(f"[Epoch {iepoch} / Batch {i}] NaN in classification loss")
            total_loss = c_loss.mean()  # + r_loss.mean() * 0.1
            # c_loss = cls_loss(predict=cls_output, target=cls_target, class_weight=None)

            # mse_loss = torch.nn.MSELoss(reduction='none')
            # c_loss = mse_loss(cls_output, torch.nn.functional.one_hot(cls_target, num_classes=9).float())
            # print(f"c_loss", c_loss.item())


            symmetric_losses = ass_loss(
                assignments=outputs["assignments"],
                detections=outputs["detections"],
                targets=batch["assignments-indices"],
                targets_mask=batch["assignments-mask"],
                process_id= None, # batch["subprocess_id"],
                num_targets=num_targets,
                event_particles=event_particles,
                event_permutations=event_info.event_permutations,
                focal_gamma=0.1,
                particle_balance=particle_balance_dict,
                process_balance=None # normalization_dict["subprocess_balance"]
            )

            assignment_predict = dict()
            ass_target, ass_mask = convert_target_assignment(
                targets = batch["assignments-indices"],
                targets_mask=batch["assignments-mask"],
                event_particles = event_particles,
                num_targets = num_targets
            )
            for process in global_config.event_info.process_names:

                total_loss += symmetric_losses["assignment"][process]
                total_loss += symmetric_losses["detection"][process]

                assignment_predict[process] = predict(
                    assignments=outputs["assignments"][process],
                    detections=outputs["detections"][process],
                    product_symbolic_groups=event_info.product_symbolic_groups[process],
                    event_permutations=event_info.event_permutations[process],
                )
                assignment_metrics[process].update(
                    best_indices=assignment_predict[process]["best_indices"],
                    assignment_probabilities=assignment_predict[process]["assignment_probabilities"],
                    detection_probabilities=assignment_predict[process]["detection_probabilities"],
                    truth_indices=ass_target[process],
                    truth_masks=ass_mask[process],
                    inputs=batch["x"],
                    inputs_mask=batch["x_mask"],
                )

            generation_loss = dict()
            for generation_target, generation_result in outputs["generations"].items():
                generation_loss[generation_target] = gen_loss(generation_result["vector"], generation_result["truth"])
                total_loss += generation_loss[generation_target]


            print(f"[Epoch {iepoch} / Batch {i}] Total loss: {total_loss}", flush=True)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"[Epoch {iepoch} / Batch {i}] Done")
            if wandb_enable:
                wandb.log(
                    {
                        "epoch": iepoch,
                        "total_loss": total_loss.item(),
                        "classification_loss": c_loss.mean().item(),
                        # "regression_loss": r_loss.mean().item()
                    }
                )
                for process in global_config.event_info.process_names:
                    wandb.log(
                        {
                            f"assignment_loss/{process}": symmetric_losses["assignment"][process].item(),
                            f"detection_loss/{process}": symmetric_losses["detection"][process].item()
                        }
                    )
                for generation_target in generation_loss:
                    wandb.log(
                        {
                            f"generation_loss/{generation_target}": generation_loss[generation_target].item()
                        }
                    )

            confusion_accumulator.update(
                cls_target,
                cls_output
            )

    if wandb_enable:
        fig = confusion_accumulator.plot_cm(class_names=num_classes)
        wandb.log({"confusion_matrix": wandb.Image(fig)})

        plt.close(fig)
        for process in assignment_metrics:
            assignment_metrics[process].assign_train_result(
                assignment_metrics[process].predict_metrics_correct,
                assignment_metrics[process].predict_metrics_wrong,
            )
            figs = assignment_metrics[process].plot_mass_spectrum()
            wandb.log({
                f"assignment_matrix/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

            figs = assignment_metrics[process].plot_score(target = "detection_score")
            wandb.log({
                f"assignment_reco_detection/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

            figs = assignment_metrics[process].plot_score(target = "assignment_score")
            wandb.log({
                f"assignment_reco_score/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

            assignment_metrics[process].reset()
    confusion_accumulator.reset()

    # break

if debugger_enable:
    debugger.remove_hooks()
    debugger.dump_debug_data()