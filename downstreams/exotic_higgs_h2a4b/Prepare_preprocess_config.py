import os, sys
sys.path.append(os.path.abspath("../.."))

import yaml
import os

import argparse
from copy import deepcopy
from pathlib import Path
import json
from collections import OrderedDict

masses = [30, 40, 60]

# Custom YAML loader/dumper that preserves key order
def ordered_yaml():
    class OrderedLoader(yaml.SafeLoader):
        pass

    class OrderedDumper(yaml.SafeDumper):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    OrderedDumper.add_representer(OrderedDict, dict_representer)

    return OrderedLoader, OrderedDumper


OrderedLoader, OrderedDumper = ordered_yaml()

def prepare_script(args):

    with open(args.config_workflow) as f:
        control = yaml.safe_load(f)

    config_farm = os.path.abspath(args.farm)
    os.makedirs(config_farm, exist_ok=True)

    working_dir = os.path.abspath(control['working_dir'])
    config_dir = os.path.abspath(os.path.dirname(args.config_workflow))
    config_file = os.path.abspath(args.config_workflow)
    cwd = os.getcwd()

    train_directory = Path(os.path.join(args.store_dir, "train"))
    train_directories = [p for p in train_directory.iterdir() if p.is_dir()]
    test_directory = Path(os.path.join(args.store_dir, "test"))
    test_directories = [p for p in test_directory.iterdir() if p.is_dir()]

    for split in ['train', 'test']:
        farm = os.path.join(config_farm, split)
        os.makedirs(farm, exist_ok=True)
        for mass in masses:
            preprocess_template = f'{cwd}/preprocess_{mass}.yaml'
            preprocess_out = os.path.join(farm, f'preprocess_{mass}.yaml')
            process_info_out = os.path.join(farm, f'process_info_{mass}.yaml')

            with open(preprocess_template, 'r') as f:
                preprocess_config = yaml.load(f, Loader=OrderedLoader)

            with open(preprocess_config["process_info"]["default"], 'r') as f:
                process_info = yaml.load(f, Loader=OrderedLoader)

            with open(os.path.join(args.store_dir, "split_summary.json"), 'r') as f:
                split_summary = json.load(f)

            for process in process_info:
                if process in split_summary:
                    process_info[process]["total_entries"] = split_summary[process][f"{split}_entries"]

            with open(process_info_out, 'w') as f:
                yaml.dump(process_info, f, Dumper=OrderedDumper)


            with open(preprocess_out, 'w') as f:
                preprocess_config['event_info']['default'] = os.path.join(cwd, preprocess_config['event_info']['default'].replace("MASS", str(mass)))
                preprocess_config['resonance']['default'] = os.path.join(cwd, preprocess_config['resonance']['default'].replace("MASS", str(mass)))
                preprocess_config['process_info']['default'] = os.path.abspath(process_info_out)
                preprocess_config['veto_double_assign'] = (split == 'train')
                yaml.dump(preprocess_config, f, Dumper=OrderedDumper)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type=str, default="config_workflow.yaml", help="Path to the workflow configuration file")
    parser.add_argument("--store_dir", type=str, default="store", help="Directory to store the output files")
    parser.add_argument("--farm", type=str, default="config_farm", help="Directory to store the configuration files")

    # Parse command-line arguments
    args = parser.parse_args()
    prepare_script(args)

if __name__ == "__main__":
    main()
