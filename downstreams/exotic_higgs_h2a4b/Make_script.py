import os, sys
sys.path.append(os.path.abspath("../.."))

import yaml
import os

import argparse
from copy import deepcopy
from pathlib import Path



def prepare_script(args):

    with open(args.config_workflow) as f:
        control = yaml.safe_load(f)



    config_farm = os.path.abspath(args.farm)
    os.makedirs(config_farm, exist_ok=True)


    working_dir = os.path.abspath(control['working_dir'])
    spanet_dir = os.path.abspath(control['spanet_dir'])
    config_dir = os.path.abspath(os.path.dirname(args.config_workflow))
    config_file = os.path.abspath(args.config_workflow)
    cwd = os.getcwd()

    os.chdir(config_dir)
    with open(control['train_yaml']) as f:
        config_template = yaml.safe_load(f)
    with open(control['predict_yaml']) as f:
        predict_template = yaml.safe_load(f)

    process_json = os.path.abspath(control['process_json'])
    stat_yml = os.path.abspath(control['stat_yml'])

    indir = control['input_dir']

    # ============= Candidates =============

    masses = control['mass_choice']
    pretrain_choice = control['pretrain_choice']
    assignment_seg_choice = control['assign_seg_choice']
    dataset_size_choice = control['dataset_size_choice']

    # ============= For debug =============
    #
    # masses = [30]
    # pretrain_choice = [True]
    # assignment_choice = [True]
    # dataset_size_choice = [0.01]

    os.makedirs(os.path.join(args.store_dir, "logs"), exist_ok=True)


    # with open(os.path.join(config_farm, "train.sh"), 'w') as f:
    #     f.write(f'#!/usr/bin/bash\n')
    #     f.write(f"#SBATCH -q {control['job_flavor']}\n")
    #     f.write(f"#SBATCH --nodes=1\n")
    #     f.write(f"#SBATCH --ntasks-per-node=1\n")
    #     f.write(f"#SBATCH --constraint=gpu\n")
    #     f.write(f"#SBATCH --gpus-per-task=4\n")
    #     f.write(f"#SBATCH -c 32\n")
    #     f.write(f'#SBATCH -A{control["account"]}\n')
    #     f.write(f"#SBATCH --time {control['time']}\n")
    #     f.write(f"#SBATCH --gpu-bind=none\n")
    #     f.write(f"#SBATCH --mail-type=END,FAIL,BEGIN\n")
    #     f.write(f'#SBATCH --mail-user={control["email"]}\n')
    #     f.write(f'#SBATCH --output={args.store_dir}/logs/job_%A_%a.out\n')
    #     f.write(f"#SBATCH --cpus-per-task=128\n")
    #     f.write(f"#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.1\n")
    #     f.write("\n")
    #
    #     narray = len(masses) * len(pretrain_choice) * len(assignment_seg_choice)
    #     f.write(f"#SBATCH --array=0-{narray-1}\n")
    #
    #
    #
    #     mass_string = ' '.join([f'"{mass}"' for mass in masses])
    #     f.write(f'masses=({mass_string})\n')
    #     pretrain_string = ' '.join([f'"pretrain"' if pretrain else '"scratch"' for pretrain in pretrain_choice])
    #     f.write(f'pretrain_choice=({pretrain_string})\n')
    #     assignment_string = ' '.join([f'"assignment-on"' if assignment else '"assignment-off"' for assignment in assignment_choice])
    #     f.write(f'assignment_choice=({assignment_string})\n')
    #     dataset_size_string = ' '.join([f'"{dataset_size}"' for dataset_size in dataset_size_choice])
    #     f.write(f'dataset_size_choice=({dataset_size_string})\n')
    #
    #     f.write(f'n_masses=${{#masses[@]}}\n')
    #     f.write(f'n_pretrain=${{#pretrain_choice[@]}}\n')
    #     f.write(f'n_assignment=${{#assignment_choice[@]}}\n')
    #
    #     f.write('mass_idx=$((SLURM_ARRAY_TASK_ID / (n_pretrain * n_assignment) % n_masses))\n')
    #     f.write('pretrain_idx=$((SLURM_ARRAY_TASK_ID / n_assignment % n_pretrain))\n')
    #     f.write('assignment_idx=$((SLURM_ARRAY_TASK_ID % n_assignment))\n')
    #
    #     f.write('mass=${masses[$mass_idx]}\n')
    #     f.write('pretrain=${pretrain_choice[$pretrain_idx]}\n')
    #     f.write('assignment=${assignment_choice[$assignment_idx]}\n')
    #
    #     f.write(f'echo "Running job for mass: ${{mass}}, pretrain: ${{pretrain}}, assignment: ${{assignment}}" \n')
    #     f.write(f'cd {working_dir}\n')
    #     f.write(f'source src.sh\n')
    #     f.write(f'cd NERSC\n')
    #
    #     f.write("head_node=$(hostname)\n")
    #     f.write("head_node_ip=$(hostname --ip-address)\n")
    #
    #     f.write('''if [[ "$head_node_ip" == *" "* ]]; then\n''')
    #     f.write('''IFS=' ' read -ra ADDR <<<"$head_node_ip"\n''')
    #     f.write("if [[ ${#ADDR[0]} -gt 16 ]]; then\n")
    #     f.write("head_node_ip=${ADDR[1]}\n")
    #     f.write("else\n")
    #     f.write("head_node_ip=${ADDR[0]}\n")
    #     f.write("fi\n")
    #     f.write("fi\n")
    #     f.write("port=6379\n")
    #     f.write(''' echo "STARTING JOB on $head_node with IP $head_node_ip"\n''')
    #     f.write('srun --nodes=1 --ntasks=1 --gpus=0 --gpus-per-task=4 --cpus-per-task=128 -w $head_node\n')
    #     f.write('shifter ./start-head.sh $head_node_ip &\n')
    #     f.write('sleep 10\n')
    #
    #     f.write(f'cd {working_dir}\n')
    #     for dataset_size in dataset_size_choice:
    #         if dataset_size < 0.2:
    #             f.write(f'shifter python3 evenet/train.py {config_farm}/evenet-ma${{mass}}-${{pretrain}}-${{assignment}}-dataset_size{dataset_size}.yaml --ray_dir {args.ray_dir} --load_all\n')
    #         else:
    #             f.write(f'shifter python3 evenet/train.py {config_farm}/evenet-ma${{mass}}-${{pretrain}}-${{assignment}}-dataset_size{dataset_size}.yaml --ray_dir {args.ray_dir}\n')
    #         f.write(f"shifter python3 evenet/predict.py {config_farm}/evenet-ma${{mass}}-${{pretrain}}-${{assignment}}-dataset_size{dataset_size}_predict.yaml \n")

    # with open(os.path.join(config_farm, "train-spanet.sh"), 'w') as f:
    #     f.write(f'#!/usr/bin/bash\n')
    #     f.write(f"#SBATCH -q {control['job_flavor']}\n")
    #     f.write(f"#SBATCH --nodes=1\n")
    #     f.write(f"#SBATCH --ntasks-per-node=4\n")
    #     f.write(f"#SBATCH --constraint=gpu\n")
    #     f.write(f"#SBATCH --gpus-per-node=4\n")
    #     f.write(f"#SBATCH -c 32\n")
    #     f.write(f'#SBATCH -A{control["account"]}\n')
    #     f.write(f"#SBATCH --time {control['time']}\n")
    #     f.write(f"#SBATCH --gpu-bind=none\n")
    #     f.write(f"#SBATCH --mail-type=END,FAIL,BEGIN\n")
    #     f.write(f'#SBATCH --mail-user={control["email"]}\n')
    #     f.write(f'#SBATCH --output={args.store_dir}/logs/job_%A_%a.out\n')
    #     f.write(f"#SBATCH --image=registry.nersc.gov/m2616/avencast/evenet:1.1\n")
    #     narray = len(masses) * len(assignment_choice)
    #     f.write(f"#SBATCH --array=0-{narray - 1}\n")
    #     f.write("\n")
    #
    #
    #     mass_string = ' '.join([f'"{mass}"' for mass in masses])
    #     f.write(f'masses=({mass_string})\n')
    #     assignment_string = ' '.join(
    #         [f'"true"' if assignment else '"false"' for assignment in assignment_choice])
    #     f.write(f'assignment_choice=({assignment_string})\n')
    #
    #     f.write(f'n_masses=${{#masses[@]}}\n')
    #     f.write(f'n_assignment=${{#assignment_choice[@]}}\n')
    #
    #     f.write('mass_idx=$((SLURM_ARRAY_TASK_ID / n_assignment % n_masses))\n')
    #     f.write('assignment_idx=$((SLURM_ARRAY_TASK_ID % n_assignment))\n')
    #
    #     f.write('mass=${masses[$mass_idx]}\n')
    #     f.write('assignment=${assignment_choice[$assignment_idx]}\n')
    #
    #     f.write(f'echo "Running job for mass: ${{mass}}, pretrain: scratch, assignment: ${{assignment}}" \n')
    #     f.write(f'cd {spanet_dir}\n')
    #     f.write(f'export SLURM_CPU_BIND="cores"\n')
    #     f.write("export MASTER_ADDR=$(hostname)\n")
    #     f.write("export MASTER_PORT=29500\n")
    #     f.write("export WORLD_SIZE=4  # total number of processes (e.g., 2 nodes)\n")
    #     f.write("export RANK=$SLURM_PROCID  # This will be unique for each process\n")
    #     f.write("export LOCAL_RANK=$SLURM_LOCALID  # This will be specific for each GPU in a node\n")
    #     f.write(f'module load conda\n')
    #     f.write(f'conda activate ray\n')
    #
    #     for dataset_size in dataset_size_choice:
    #         f.write(f'dataset={args.store_dir}/spanet-train/spanet-ma${{mass}}/data.h5\n')
    #
    #         # Write if-else in shell script to choose options_file and assignment_flag
    #         f.write('if [ "${assignment}" = "true" ]; then\n')
    #         f.write('    options_file="options_files/exotic_higgs_decay/full_training.json"\n')
    #         f.write('    assignment_flag="-on"\n')
    #         f.write('else\n')
    #         f.write('    options_file="options_files/exotic_higgs_decay/full_training-cls.json"\n')
    #         f.write('    assignment_flag="-off"\n')
    #         f.write('fi\n')
    #
    #         run_name = f'spanet-ma${{mass}}-scratch-assignment${{assignment_flag}}-dataset_size{dataset_size}'
    #         log_dir = args.store_dir
    #
    #         f.write(f'run_name="{run_name}"\n')
    #         f.write(f'log_dir="{log_dir}"\n')
    #
    #         epochs = 50 if dataset_size > 0.1 else 100
    #
    #         if dataset_size > 0.09:
    #             batch_size = 2048
    #         elif dataset_size > 0.02:
    #             batch_size = 1024
    #         else:
    #             batch_size = 512
    #
    #         f.write(
    #             f'srun python3 -m spanet.train --event_file event_files/haa_ma${{mass}}.yaml '
    #             f'-tf ${{dataset}} --options_file ${{options_file}} --log_dir ${{log_dir}} '
    #             f'--run_name ${{run_name}} --epochs {epochs} --gpus 4 -b {batch_size} --limit_dataset {dataset_size * 100} '
    #             f'--project {control["spanet"]["project"]}\n'
    #         )
    #         # f.write(
    #         #     f'srun python3 -m spanet.predict ${{log_dir}}/checkpoints/${{run_name}} '
    #         #     f'{args.store_dir}/predictions/${{run_name}}/predict.h5 -tf '
    #         #     f'{args.store_dir}/spanet-test/spanet-ma${{mass}}/data.h5 '
    #         #     f'--event_file event_files/haa_ma${{mass}}.yaml --batch_size 2048 --gpu\n\n'
    #         # )
    with open(os.path.join(config_farm, "prepare-dataset.sh"), 'w') as f:
        f.write(f'python3 Split_dataset.py {" ".join(indir)} --output_dir {args.store_dir}\n')
        f.write(f'python3 Prepare_preprocess_config.py {config_file} --store_dir {args.store_dir} --farm {config_farm}\n')
        for mass in masses:
            f.write(f"cd {working_dir}\n")
            for split in ["train", "test"]:
                # pretrain_dir = Path(os.path.join(args.store_dir, split))
                # pretrain_dir = [str(p) for p in pretrain_dir.iterdir() if p.is_dir()]
                f.write(f"shifter python3 preprocessing/preprocess.py {config_farm}/{split}/preprocess_{mass}.yaml --store_dir {args.store_dir}/evenet-{split}/evenet-ma{mass} --pretrain_dirs {args.store_dir}/{split}/Combined_Balanced \n")
                f.write(f"shifter python3 preprocessing/convert_evenet_to_spanet.py {cwd}/configs/event_info_{mass}.yaml --in_dir {args.store_dir}/evenet-{split}/evenet-ma{mass} --store_dir {args.store_dir}/spanet-{split}/spanet-ma{mass}\n")
            for pretrain in pretrain_choice:
                for assignment, segmentation in assignment_seg_choice:
                    for dataset_size in dataset_size_choice:
                        os.chdir(config_dir)
                        config = deepcopy(config_template)
                        config['network']['default'] = os.path.abspath(config['network']['default'])
                        config['event_info']['default'] =  os.path.abspath(config['event_info']['default'].replace("MASS", str(mass)))
                        config['resonance']['default'] = os.path.abspath(config['resonance']['default'])
                        config['options']['default'] = os.path.abspath(config['options']['default'])
                        config['logger']['wandb']['run_name'] = f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
                        config['options']['Dataset']['normalization_file'] = os.path.join(f"{args.store_dir}/evenet-train/evenet-ma{mass}","normalization.pt")
                        config['options']['Dataset']['dataset_limit'] = dataset_size
                        config['platform']["data_parquet_dir"] =  f"{args.store_dir}/evenet-train/evenet-ma{mass}"
                        if assignment:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['assignment'] = [1.0, 1.0]
                            config["options"]["Training"]["Components"]["Assignment"]['include'] = True
                        else:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['assignment'] = [0.0, 0.0]
                            config["options"]["Training"]["Components"]["Assignment"]['include'] = False


                        if segmentation:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['segmentation'] = [1.0, 1.0]
                            config["options"]["Training"]["Components"]["Segmentation"]['include'] = True
                        else:
                            config["options"]["Training"]["ProgressiveTraining"]["stages"][0]['loss_weights']['segmentation'] = [0.0, 0.0]
                            config["options"]["Training"]["Components"]["Segmentation"]['include'] = False


                        if dataset_size < 0.1:
                            config["options"]["Training"]["epochs"] = 100
                            config["options"]["Training"]["total_epochs"] = 100

                        config["options"]["Training"]["model_checkpoint_save_path"] = os.path.join(args.store_dir, "checkpoints", f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')
                        if not pretrain:
                            config["options"]["Training"]["pretrain_model_load_path"] = None
                        else:
                            config['options']['default'] = os.path.abspath(config['options']['default']).replace(".yaml", "_pretrain.yaml")

                        predict_config = deepcopy(predict_template)
                        predict_config["platform"]["data_parquet_dir"] = f"{args.store_dir}/evenet-test/evenet-ma{mass}"
                        predict_config["options"]["default"] = os.path.abspath(predict_config["options"]["default"])
                        predict_config["options"]["prediction"]["output_dir"] = os.path.join(args.store_dir, "predictions", f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')

                        if segmentation:
                            predict_config["options"]["Training"]["Components"]["Segmentation"]['include'] = True
                        else:
                            predict_config["options"]["Training"]["Components"]["Segmentation"]['include'] = False

                        if assignment:
                            predict_config["options"]["Training"]["Components"]["Assignment"]['include'] = True
                        else:
                            predict_config["options"]["Training"]["Components"]["Assignment"]['include'] = False



                        ckpt_dir =  os.path.join(args.store_dir, "checkpoints", f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')

                        predict_config["options"]["Training"]["model_checkpoint_load_path"] = ckpt_dir
                        predict_config["options"]["Dataset"]["normalization_file"] = os.path.join(f"{args.store_dir}/evenet-train/evenet-ma{mass}", "normalization.pt")

                        predict_config["network"]["default"] = os.path.abspath(predict_config["network"]["default"])
                        predict_config["event_info"]["default"] = os.path.abspath(predict_config["event_info"]["default"].replace("MASS", str(mass)))
                        predict_config["resonance"]["default"] = os.path.abspath(predict_config["resonance"]["default"])

                        file_path = os.path.join(config_farm, f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}.yaml')
                        file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}_predict.yaml')
                        os.chdir(cwd)
                        with open(file_path, 'w') as fout:
                            yaml.dump(config, fout)
                        with open(file_path_predict, 'w') as fout:
                            yaml.dump(predict_config, fout)

    with open(os.path.join(config_farm, "train-evenet.sh"), 'w') as f:
        # f.write(f'python3 Split_dataset.py {" ".join(indir)} --output_dir {args.store_dir}\n')
        # f.write(f'python3 Prepare_preprocess_config.py {config_file} --store_dir {args.store_dir} --farm {config_farm}\n')
        for mass in masses:
            for pretrain in pretrain_choice:
                for assignment, segmentation in assignment_seg_choice:
                    for dataset_size in dataset_size_choice:
                        file_path = os.path.join(config_farm, f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}.yaml')
                        file_path_predict = os.path.join(config_farm, f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}_predict.yaml')
                        os.chdir(cwd)
                        job_name = f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}'

                        f.write(f"cd {working_dir} && ")
                        f.write(f"python3 evenet/train.py {file_path} --ray_dir {args.ray_dir} {'--load_all' if dataset_size < 0.2 else ''} && ")
                        f.write(f"python3 evenet/predict.py {os.path.abspath(file_path_predict)} \n")

    with open(os.path.join(config_farm, "summary.sh"), 'w') as f:
        f.write(f"cd {cwd}\n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir}\n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir} --network spanet\n")

        for pretrain in pretrain_choice:
            for assignment, segmentation in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    store_directory = os.path.join(args.store_dir, "ntuple",
                                                   f'evenet-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')
                    out_directory = os.path.join(args.store_dir, "fit",
                                                   f'evenet-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')
                    f.write(f"python3 Statistics_test.py --Lumi {args.Lumi} --signal all --process_json {process_json} --sourceFile {store_directory}/ntuple.root --observable MVAscoreMASS --config_yml {stat_yml} --outdir {out_directory} --log_scale & \n")
        f.write(f"python3 Produce_ntuple.py {config_file} --store_dir {args.store_dir} --network spanet\n")
        for pretrain in [False]:
            for assignment, _ in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    store_directory = os.path.join(args.store_dir, "ntuple",
                                                   f'spanet-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}')
                    out_directory = os.path.join(args.store_dir, "fit",
                                                   f'spanet-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}')
                    f.write(f"python3 Statistics_test.py --Lumi {args.Lumi} --signal all --process_json {process_json} --sourceFile {store_directory}/ntuple.root --observable MVAscoreMASS --config_yml {stat_yml} --outdir {out_directory} --log_scale & \n")
        f.write(f"python3 Summary_Limit.py --store_dir {args.store_dir}\n")

    with open(os.path.join(config_farm, "train_spanet.sh"), 'w') as f:
        for mass in masses:
            f.write(f"cd {spanet_dir}\n")
            for assignment, _ in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    dataset_dir = f"{args.store_dir}/spanet-train/spanet-ma{mass}"
                    dataset = f"{dataset_dir}/data.h5"
                    run_name = f'spanet-ma{mass}-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
                    options_file = "options_files/exotic_higgs_decay/full_training.json" if assignment else "options_files/exotic_higgs_decay/full_training-cls.json"
                    log_dir = os.path.join(args.store_dir)

                    epochs = 100 if dataset_size < 0.1 else 50

                    if dataset_size > 0.09:
                        batch_size = 2048
                    elif dataset_size > 0.02:
                        batch_size = 1024
                    else:
                        batch_size = 512

                    f.write(
                        f"python3 -m spanet.train --event_file event_files/haa_ma{mass}.yaml -tf {dataset} --options_file {options_file} --log_dir {log_dir} --run_name {run_name} --epochs {epochs} --gpus 4 --limit_dataset {dataset_size * 100} -b {batch_size} --project {control['spanet']['project']} \n")


    with open(os.path.join(config_farm, "predict_spanet.sh"), 'w') as f:
        for mass in masses:
            f.write(f"cd {spanet_dir}\n")
            for assignment, _ in assignment_seg_choice:
                for dataset_size in dataset_size_choice:
                    dataset_dir = f"{args.store_dir}/spanet-train/spanet-ma{mass}"
                    dataset = f"{dataset_dir}/data.h5"
                    run_name = f'spanet-ma{mass}-scratch-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}'
                    options_file =  "options_files/exotic_higgs_decay/full_training.json" if assignment else "options_files/exotic_higgs_decay/full_training-cls.json"
                    log_dir = os.path.join(args.store_dir)
#                    f.write(f"python3 -m spanet.train --event_file event_files/haa_ma{mass}.yaml -tf {dataset} --options_file {options_file} --log_dir {log_dir} --run_name {run_name} --epochs 50 --gpus 4 --limit_dataset {dataset_size * 100} --project {control['spanet']['project']} \n")
                    f.write(f"python3 -m spanet.predict {log_dir}/checkpoints/{run_name} {args.store_dir}/predictions/{run_name}/predict.h5 -tf {args.store_dir}/spanet-test/spanet-ma{mass}/data.h5  --event_file event_files/haa_ma{mass}.yaml --batch_size 1024 --gpu\n")
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type=str, default="config_workflow.yaml", help="Path to the workflow configuration file")
    parser.add_argument("--store_dir", type=str, default="store", help="Directory to store the output files")
    parser.add_argument("--ray_dir", type=str, default="ray", help="Directory for Ray cluster")
    parser.add_argument("--farm", type=str, default="config_farm", help="Directory to store the configuration files")
    parser.add_argument("--Lumi", type=float, default=1000.0, help="Luminosity for the simulation")

    # Parse command-line arguments
    args = parser.parse_args()
    prepare_script(args)

if __name__ == "__main__":
    main()
