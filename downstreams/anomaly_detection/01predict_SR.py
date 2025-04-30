import argparse
import yaml
import os

import json
import numpy as np
import pyarrow.parquet as pq
import awkward as ak
import vector
vector.register_awkward()
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.diffusion_sampler import DDIMSampler


from helpers.basic_fit import plot_mass_distribution
from helpers.flow_sampling import get_mass_samples
from helpers.stats_functions import curve_fit_m_inv, parametric_fit, check_bkg_for_peaks
from helpers.plotting import read_feature
from helpers.utils import save_file

from rich.progress import Progress, BarColumn, TimeRemainingColumn

from preprocessing.preprocess import unflatten_dict
from functools import partial


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Choose a free port
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def clean_and_append(dirname, postfix):
    if dirname.endswith("/"):
        dirname = dirname[:-1]
    return dirname + postfix

def predict(rank, world_size, gen_num_events, args):

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    seed = 42 + rank  # Adding rank to the seed for uniqueness
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Rank {rank}] Starting prediction on device {device}")


    postfix = "" if not args.no_signal else "_no_signal"
    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    step_dir = f"step1_generate_SR{postfix}"
    inputdir = clean_and_append(config["output"]["storedir"], postfix)



    os.makedirs(os.path.join(config["output"]["plotdir"], step_dir), exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    global_config.load_yaml(config["train"]["config"])
    normalization_dict = torch.load(os.path.join(inputdir, "SB", "normalization.pt"))
    shape_metadata = json.load(open(os.path.join(inputdir, "SB", "shape_metadata.json")))
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)

    model = EveNetModel(
        config = global_config,
        device = torch.device(device),
        normalization_dict=normalization_dict,
        point_cloud_generation=True
    ).to(device)

    new_state_dict = dict()
    for k, v in checkpoint["state_dict"].items():
        new_state_dict[k.replace("model.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    df_dummy_SB = pq.read_table(
        os.path.join(inputdir, "SB", "data.parquet")
    ).to_pandas()
    df_dummy_SB.sample(frac=1).reset_index(drop=True)
    df_dummy_SR = pq.read_table(
        os.path.join(inputdir, "SR", "data.parquet")
    ).to_pandas()
    df_dummy_SR.sample(frac=1).reset_index(drop=True)
    norm_dict = torch.load(os.path.join(inputdir, "SB", "normalization.pt"))


    df_SB = process_event_batch(
        df_dummy_SB,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    df_SR = process_event_batch(
        df_dummy_SR,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    inv_mass_truth = np.concatenate([df_SB["conditions"][..., 0], df_SR["conditions"][..., 0]], axis=0)

    inv_mass_results = plot_mass_distribution(
        inv_mass_truth,
        SR_left=config['mass-windows']['SR-left'],
        SR_right=config['mass-windows']['SR-right'],
        SB_left=config['mass-windows']['SB-left'],
        SB_right=config['mass-windows']['SB-right'],
        bkg_fit_degree=config['fit']['bkg-fit-degree'],
        num_bins_SR=config['mass-windows']['SR-bins'],
        save_name=os.path.join(config['output']['plotdir'], step_dir, "truth_mass_distribution.png")
    )

    # Get the data for the model
    mass_sample = get_mass_samples(
        SR_left=config["mass-windows"][f"{args.region}-left"],
        SR_right=config["mass-windows"][f"{args.region}-right"],
        n_SR_samples = gen_num_events,
        popt = inv_mass_results["popt_0"],
        fit_function = parametric_fit
    )

    mass_sample = torch.tensor(mass_sample, dtype=torch.float32).to(device).unsqueeze(-1)

    num_generated_sample = 0
    dummy_batch = convert_batch_to_torch_tensor(df_SR)
    dummy_batch = {
        k: v[:args.batch_size].to(device) for k, v in dummy_batch.items()
    }
    # Initialize the progress bar

    sampler = DDIMSampler(device)
    gen_sample = {}

    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Generating samples...", total=gen_num_events)

        while num_generated_sample < gen_num_events:
            batch_num = max(min(gen_num_events - num_generated_sample, args.batch_size) , 1)

            # Your generation logic here
            # ...

            condition_x = {k: v.clone()[:batch_num] for k, v in dummy_batch.items()}
            condition_x["conditions"] = mass_sample[num_generated_sample:num_generated_sample + batch_num]

            predict_for_point_cloud = partial(
                model.predict_diffusion_vector,
                mode="event",
                cond_x=condition_x,
                noise_mask=condition_x["x_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )  # TODO: add stuff from previous step.

            generated_distribution = sampler.sample(
                data_shape=condition_x["x"].shape,
                pred_fn=predict_for_point_cloud,
                normalize_fn=model.sequential_normalizer,
                eta=1.0,
                num_steps=args.num_steps
            )
            num_generated_sample += batch_num
            progress.update(task, advance=batch_num)

            for key in condition_x:
                if key not in gen_sample:
                    gen_sample[key] = []
                else:
                    if "x" == key:
                        gen_sample[key].append(generated_distribution)
                    else:
                        gen_sample[key].append(condition_x[key])



    gen_sample = {k: torch.cat(v, dim=0).detach().cpu().numpy() for k, v in gen_sample.items()}
    jet = ak.from_regular(vector.zip(
        {
            "pt": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'pt')),
            "eta": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'eta')),
            "phi": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'phi')),
            "mass": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'mass')),
            "MASK": ak.from_numpy(gen_sample['x_mask'])
        }
    ))

    inv_mass_gen = (jet[..., 0] + jet[..., 1]).mass
    _ = plot_mass_distribution(
        inv_mass_gen,
        SR_left=config['mass-windows']['SR-left'],
        SR_right=config['mass-windows']['SR-right'],
        SB_left=config['mass-windows']['SB-left'],
        SB_right=config['mass-windows']['SB-right'],
        bkg_fit_degree=config['fit']['bkg-fit-degree'],
        num_bins_SR=config['mass-windows']['SR-bins'],
        save_name=os.path.join(config['output']['plotdir'], step_dir, f"gen_mass_distribution_{args.region}.png")
    )

    gen_sample["conditions"] = inv_mass_gen.to_numpy().reshape(-1, 1)

    SR_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SR-left']) & (
                inv_mass_gen.to_numpy() < config['mass-windows']['SR-right'])
    SB_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SB-left']) & (
                inv_mass_gen.to_numpy() < config['mass-windows']['SB-right'])
    SB_filter = SB_filter & ~SR_filter
    if args.region == "SR":
        gen_sample = {k: v[SR_filter] for k, v in gen_sample.items()}
    else:
        gen_sample = {k: v[SB_filter] for k, v in gen_sample.items()}


    if args.region == "SR":
        df_data = df_SR
    else:
        df_data = df_SB

    gen_sample["classification"] = np.zeros_like(gen_sample["classification"])
    df_data["classification"] = np.ones_like(df_data["classification"])

    num_gen = gen_sample["x"].shape[0]
    num_data = df_data["x"].shape[0]
    num_total = num_gen + num_data


    norm_dict['class_counts'] = torch.tensor([num_gen, num_data], dtype=torch.float32)
    norm_dict['class_balance'] = torch.tensor([num_total/num_gen, num_total/num_data], dtype=torch.float32)
    norm_dict['subprocess_counts'] = norm_dict['class_counts']
    norm_dict['subprocess_balance'] = norm_dict['class_balance']

    # df_hybrid = {k: np.concatenate([gen_sample[k], df_data[k]], axis=0) for k in gen_sample}
    # df_hybrid["conditions"] = np.ones_like(df_hybrid["conditions"]) # Remove the mass condition

    # perm = np.random.permutation(len(next(iter(df_hybrid.values()))))
    # df_hybrid = {k: v[perm] for k, v in df_hybrid.items()}

    outputdir = clean_and_append(config["output"]["storedir"], "_gen")

    save_file(
        save_dir = os.path.join(outputdir, f"{args.region}"),
        data_df = gen_sample,
        norm_dict = norm_dict,
        event_filter = None,
        postfix = f"_fold{rank}",
    )
    cleanup()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--checkpoint", type = str)
    parser.add_argument("--gen_num_events", type = int, default = 10240)
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--num_steps", type = int, default = 20)
    parser.add_argument("--region", type = str, default = "SR")
    parser.add_argument("--no_signal", action = "store_true", default = False)
    parser.add_argument("--ngpu", type = int, default = 1)
    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file

    world_size = args.ngpu
    print(f"Using {world_size} GPUs for distributed eval.")
    mp.spawn(
        predict,
        args=(world_size, int(args.gen_num_events / world_size), args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
