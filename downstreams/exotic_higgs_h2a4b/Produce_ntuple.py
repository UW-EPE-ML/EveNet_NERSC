import os, sys
sys.path.append(os.path.abspath("../.."))
from scipy.special import softmax

import yaml
import os
import ROOT
import array

import argparse
from copy import deepcopy
import torch
masses = [30, 40, 60]

def prepare_ntuple(args):
    for pretrain in [True, False]:
        for assignment, segmentation in [[True, False], [False, True], [False, False], [True, True]]:
            for dataset_size in [0.01, 0.03, 0.1, 0.3, 1.0]:
                hists = dict()
                for mass in masses:

                    if args.network == "evenet":
                        file_dir = os.path.join(args.store_dir, "predictions", f'evenet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')
                        fname = os.path.join(file_dir, "prediction.pt")
                        process_id_dict = {-1: "QCD", 0: "haa_maMASS"}


                    elif args.network == "spanet":
                        file_dir = os.path.join(args.store_dir, "predictions", f'spanet-ma{mass}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}-dataset_size{dataset_size}')
                        fname = os.path.join(file_dir, "predict.h5")
                        process_id_dict = {0: "QCD", 1: "haa_maMASS"}

                    if os.path.exists(fname):
                        print(f"\033[92mFile {fname} exists.\033[0m")  # Green text
                    else:
                        print(f"\033[91mFile {fname} does not exist, skipping.\033[0m")  # Red text
                        continue

                    if args.network == "evenet":
                        df = torch.load(fname, map_location='cpu')

                        mvascore = torch.concat([torch.nn.functional.softmax(data["classification"]["classification/signal"], dim=1)[...,1] for data in df], dim=0)
                        process_id = torch.concat([data["subprocess_id"] for data in df], dim=0)
                        event_weight = torch.concat([data["event_weight"] for data in df], dim=0)

                    elif args.network == "spanet":
                        import h5py
                        with h5py.File(fname, 'r') as f:
                            logits = f['SpecialKey.Classifications/EVENT/signal']
#                            mvascore = torch.tensor(softmax(logits[:], axis=1)[:, 1])
                            mvascore = torch.tensor(logits[:, 1])
                            process_id = torch.tensor(f['SpecialKey.Inputs/SpecialKey.Classifications/signal'][:])
                            event_weight = torch.tensor(f['SpecialKey.Inputs/weight'][:])

                    array_list = dict()
                    for id, process in process_id_dict.items():
                        array_list[process] = {
                            "y": mvascore[process_id == id],
                            "w": event_weight[process_id == id]
                        }


                    for process, data in array_list.items():
                        y_all = data["y"].detach().cpu().numpy()
                        w_all = data["w"].detach().cpu().numpy()

                        y = y_all
                        w = w_all
                        hist = ROOT.TH1F('h', "weighted", 1000, 0, 1)
                        hist.FillN(len(y), array.array('d', y), array.array('d', w))
                        hists[f"{process}_MVAscoreMASS_SR".replace("MASS", str(mass))] = hist

                # Create the output directory if it doesn't exist

                if len(hists) == 0:
                    print(f"\033[93mNo histograms to write for mass {mass}, skipping.\033[0m")
                    continue
                store_directory = os.path.join(args.store_dir, "ntuple", f'{args.network}-{"pretrain" if pretrain else "scratch"}-assignment{"-on" if assignment else "-off"}{"-segmentation-on" if segmentation else ""}-dataset_size{dataset_size}')
                store_root_name = os.path.join(store_directory, "ntuple.root")
                os.makedirs(store_directory, exist_ok=True)
                # Create and open a new ROOT file
                f_out = ROOT.TFile(store_root_name, "RECREATE")

                # Write histogram
                for hist_name, hist in hists.items():
                    # Create a new histogram with the same properties
                    hist = hist.Clone()
                    hist.SetName(hist_name)
                    hist.Write()
                # Close the file
                f_out.Close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type=str, default="config_workflow.yaml", help="Path to the workflow configuration file")
    parser.add_argument("--store_dir", type=str, default="store", help="Directory to store the output files")
    parser.add_argument("--network", type=str, default="evenet", help="Network name to use for predictions")
    # Parse command-line arguments
    args = parser.parse_args()
    prepare_ntuple(args)

if __name__ == "__main__":
    main()
