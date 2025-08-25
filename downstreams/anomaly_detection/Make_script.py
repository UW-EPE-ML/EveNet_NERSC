import os, sys
sys.path.append(os.path.abspath("../.."))

import yaml
import os
import vector
vector.register_awkward()
import argparse
from helpers.utils import clean_and_append


def prepare_script(args):

    with open(args.config_workflow, 'r') as f:
        control = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.makedirs(args.farm, exist_ok=True)

    with open(f"{args.farm}/run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f'python3 00prepare_CMSOpenData.py {args.config_workflow}\n')
        storedir = control["output"]["storedir"]

        cwd = os.getcwd()
        os.chdir(base_dir)
        with open(control["train"]["config"], 'r') as f_config:
            train_config = yaml.safe_load(f_config)
        train_config["platform"]["data_parquet_dir"] = f"{storedir}/SB"
        train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{storedir}/SB/model_checkpoint"
        train_config["options"]["Dataset"]["normalization_file"] = f"{storedir}/SB/normalization.pt"

        with open(control["train"]["config"], 'w') as f_config:
            yaml.dump(train_config, f_config, default_flow_style=False)

        if args.no_signal:
            no_signal_storedir = clean_and_append(storedir, "_no_signal")
            train_config["platform"]["data_parquet_dir"] = f"{no_signal_storedir}/SB"
            train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{no_signal_storedir}/SB/model_checkpoint"
            train_config["options"]["Dataset"]["normalization_file"] = f"{no_signal_storedir}/SB/normalization.pt"

            with open(control["train"]["config"].replace(".yaml", "-nosig.yaml"), 'w') as f_config:
                yaml.dump(train_config, f_config, default_flow_style=False)

        f.write(f'python3 03kfold_train.py {os.path.abspath(control["train"]["config"])} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm}-gen --local \n')
        f.write(f'cd {control["workdir"]} \n')
        f.write(f'sh /global/homes/t/tihsu/script/submit_ray_job_para.sh {cwd}/{args.farm}-gen/train.sh \n')
        f.write(f'cd {cwd} \n')

        if args.no_signal:
            f.write(f'python3 03kfold_train.py {os.path.abspath(control["train"]["config"].replace(".yaml", "-nosig.yaml"))} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm}-gen-nosignal --local \n')
            f.write(f'cd {control["workdir"]} \n')
            f.write(f'sh /global/homes/t/tihsu/script/submit_ray_job_para.sh {cwd}/{args.farm}-gen-nosignal/train.sh \n')
            f.write(f'cd {cwd} \n')

        os.chdir(cwd)

        with open(f"{args.farm}/predict.sh", "w") as f_predict:
            # Predict
            f_predict.write(f'python3 01predict_SR.py {os.path.abspath(args.config_workflow)} --region SB --checkpoint {f"{storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k}\n')
            f_predict.write(f'python3 01predict_SR.py {os.path.abspath(args.config_workflow)} --region SR --checkpoint {f"{storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k}\n')

            if args.no_signal:
                f_predict.write(f'python3 01predict_SR.py {os.path.abspath(args.config_workflow)} --region SB --checkpoint {f"{no_signal_storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k} --no_signal\n')
                f_predict.write(f'python3 01predict_SR.py {os.path.abspath(args.config_workflow)} --region SR --checkpoint {f"{no_signal_storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k} --no_signal\n')

        f.write(f'source ~/script/kill_srun.sh; sh ~/script/run_on_16_gpus_ngpu.sh --gpus-per-task {args.gpu} {os.path.abspath(args.farm)}/predict.sh \n')
        f.write(f'python3 02prepare_classification_dataset.py {args.config_workflow} --region SR --max_background {args.max_background} \n')
        f.write(f'python3 02prepare_classification_dataset.py {args.config_workflow} --region SB --max_background {args.max_background}  \n')
        f.write(f'python3 02prepare_classification_dataset.py {args.config_workflow} --region SR --no_signal --max_background {args.max_background}  \n')
        f.write(f'python3 02prepare_classification_dataset.py {args.config_workflow} --region SB --no_signal --max_background {args.max_background} \n')


        drop_str = f"--drop {' '.join(args.drop)}" if len(args.drop) > 0 else ""
        only_pc_str = "--only_pc" if args.only_pc else ""

        f.write(f'python3 03train_cls.py {args.config_workflow} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --test_no_signal\n')
        f.write(f'python3 03train_cls.py {args.config_workflow} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --no_signal --test_no_signal\n')
        #f.write(f'python3 03train_cls.py {args.config_workflow}  --knumber {args.k} {"--only_pc" if args.only_pc else ""} {f"--drop {" ".join(args.drop)}" if len(args.drop) > 0 else ""} --ignore pfn --test_no_signal\n')
        #f.write(f'python3 03train_cls.py {args.config_workflow}  --knumber {args.k} {"--only_pc" if args.only_pc else ""} {f"--drop {" ".join(args.drop)}" if len(args.drop) > 0 else ""} --ignore pfn --no_signal --test_no_signal\n')

        os.chdir(base_dir)
        with open(control["train-cls"]["config"], 'r') as f_config:
            train_config = yaml.safe_load(f_config)

        cls_store_dir = clean_and_append(storedir, "_hybrid")
        train_config["platform"]["data_parquet_dir"] = f"{cls_store_dir}/SR"
        train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{cls_store_dir}/SR/model_checkpoint"
        train_config["options"]["Dataset"]["normalization_file"] = f"{cls_store_dir}/SR/normalization.pt"
        train_config['wandb']['run_name'] = f"{train_config['wandb']['run_name']}_pretrain"
        with open(control["train-cls"]["config"], 'w') as f_config:
            yaml.dump(train_config, f_config, default_flow_style=False)



        cls_train_config_path = os.path.abspath(control["train-cls"]["config"])

        train_config["options"]["Training"]["pretrain_model_load_path"] = None
        train_config['wandb']['run_name'] = f"{train_config['wandb']['run_name']}_scratch"
        train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{cls_store_dir}/SR/model_checkpoint_scratch"


        with open(control["train-cls"]["config"].replace(".yaml", "_scratch.yaml"), 'w') as f_config:
            yaml.dump(train_config, f_config, default_flow_style=False)

        cls_train_scratch_path = os.path.abspath(control["train-cls"]["config"].replace(".yaml", "_scratch.yaml"))

        os.chdir(cwd)

        farm_dir = os.path.abspath(args.farm)
        farm_scratch_dir = os.path.abspath(args.farm + "_scratch")

        f.write(f'python3 04bump_hunting.py {os.path.abspath(args.config_workflow)}  \n')
        f.write(f'python3 04bump_hunting.py {os.path.abspath(args.config_workflow)} --test_no_signal \n')
        f.write(f'python3 04bump_hunting.py {os.path.abspath(args.config_workflow)} --test_no_signal --no_signal \n')
        f.write(f'python3 04bump_hunting.py {os.path.abspath(args.config_workflow)} --no_signal \n')

        # f.write(f'python3 03kfold_train.py {cls_train_config_path} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm} {"--local" if args.local else ""} \n')
        # f.write(f'python3 03kfold_train.py {cls_train_scratch_path} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm + "_scratch"} {"--local" if args.local else ""} \n')
        # f.write(f'cd {control["workdir"]} \n')
        # f.write(f'sh {farm_dir}/train.sh \n')
        # f.write(f'sh {farm_scratch_dir}/train.sh \n')



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--farm", type = str, default = "Farm")
    parser.add_argument("--ray_dir", type = str, default = "/pscratch/sd/t/tihsu/tmp")
    parser.add_argument("--gen_events", type = int, default = 102400)
    parser.add_argument("--gpu", type = int, default = 1)
    parser.add_argument("--max_background", type = int, default = 40000)
    parser.add_argument("--k", type = int, default = 3)
    parser.add_argument("--local", action='store_true', help="Run locally instead of on the farm")
    parser.add_argument("--only_pc", action='store_true')
    parser.add_argument("--drop", nargs = "+", type = str)
    parser.add_argument("--no_signal", action='store_true', help="Do not use signal processing features")
    parser.add_argument("--test_no_signal", action='store_true', help="Test mode for no signal processing features")
    # Parse command-line arguments
    args = parser.parse_args()

    prepare_script(args)

if __name__ == "__main__":
    main()
