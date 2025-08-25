import pickle
import os, sys
sys.path.append(os.path.abspath("../.."))

from helpers.utils import save_file

from helpers.physics_functions import assemble_m_inv, muon_mass, calculate_deltaR

import yaml
import torch
import os
import awkward as ak
import vector
vector.register_awkward()
import argparse
from helpers.plotting import *
from helpers.utils import mean_std_last_dim, clean_and_append

def create_point_cloud(data, code):
    pt = np.stack([data[code]['mu0_pt'], data[code]['mu1_pt']], axis=-1)
    eta = np.stack([data[code]['mu0_eta'], data[code]['mu1_eta']], axis=-1)
    phi = np.stack([data[code]['mu0_phi'], data[code]['mu1_phi']], axis=-1)
    mass = np.ones_like(pt) * muon_mass
    mask = np.ones_like(pt, dtype=bool)
    ip3d = np.stack([data[code]['mu0_ip3d'], data[code]['mu1_ip3d']], axis=-1)
    charge = np.stack([np.ones_like(data[code]['mu0_phi']), -1 * np.ones_like(data[code]['mu0_phi'])],
                      axis=-1)  # Artificial: charge
    isbtag = np.zeros_like(pt)
    isLepton = np.ones_like(pt)

    muon = ak.from_regular(vector.zip(
        {
            "pt": ak.from_numpy(pt),
            "eta": ak.from_numpy(eta),
            "phi": ak.from_numpy(phi),
            "mass": ak.from_numpy(mass),
            "MASK": ak.from_numpy(mask),
            "ip3d": ak.from_numpy(ip3d),
            "charge": ak.from_numpy(charge),
            "isLepton": ak.from_numpy(isLepton),
            "btag": ak.from_numpy(isbtag)
        }
    ))

    muon = muon[ak.argsort(muon.pt, ascending=False)]

    return muon


def prepare_CMSOpenData(args):



    with open(args.config_workflow, 'r') as f:
        control = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.chdir(base_dir)

    with open(control["config_workflow"], 'r') as f:
        workflow = yaml.safe_load(f)

    with open(control["input"]["event_info"], 'r') as f:
        event_info = yaml.safe_load(f)

    point_cloud_feature_to_save = event_info["INPUTS"]["SEQUENTIAL"]["Source"]
    condition_feature_to_save = event_info["INPUTS"]["GLOBAL"]["Conditions"]

    processed_data_dir = workflow["file_paths"]["data_storage_dir"] +"/projects/"+workflow["analysis_keywords"]["name"]+"/processed_data/"
    os.makedirs(processed_data_dir, exist_ok = True)

    feature_set_to_save = workflow["feature_sets"]["mix_2"]
    feature_set_to_save.append("dimu_mass")


    working_dir = workflow["file_paths"]["working_dir"]
    path_to_compiled_data = workflow["file_paths"]["data_storage_dir"] + "/compiled_data/" + workflow["analysis_keywords"][
        "dataset_id"]

    codes_list = ["skimmed_data_2016H_30555_nojet"]  # may want multiple codes for injection studies

    uncut_data_OS, uncut_data_SS = {code: {} for code in codes_list}, {code: {} for code in codes_list}

    for code in codes_list:
        with open(f"{path_to_compiled_data}/{code}", "rb") as ifile:
            tmp_dict = pickle.load(ifile)
            for key in tmp_dict.keys():
                if "samesign" in key:
                    uncut_data_SS[code][key[:-9]] = tmp_dict[key]
                else:
                    uncut_data_OS[code][key] = tmp_dict[key]

        print(code, "opp sign", uncut_data_OS[code][list(uncut_data_OS[code].keys())[0]].shape)
        print(code, "same sign", uncut_data_SS[code][list(uncut_data_SS[code].keys())[0]].shape)

    feature_set = list(uncut_data_OS[codes_list[0]].keys())
    feature_set = [x for x in feature_set if "HLT" not in x]


    SB_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_left"])
    SR_left = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_left"])
    SR_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SR_right"])
    SB_right = float(workflow["window_definitions"][workflow["analysis_keywords"]["particle"]]["SB_right"])

    band_bounds = {"SBL": [SB_left, SR_left],
                   "SR": [SR_left, SR_right],
                   "SBH": [SR_right, SB_right],
                   }

    print(band_bounds)

    cut_data_OS, cut_data_SS = {code: {} for code in codes_list}, {code: {} for code in codes_list}

    analysis_cuts_dict = workflow["analysis_keywords"]["analysis_cuts"]

    for code in codes_list:

        pass_indices_OS = np.ones((uncut_data_OS[code]["dimu_mass"].shape[0]))
        pass_indices_SS = np.ones((uncut_data_SS[code]["dimu_mass"].shape[0]))
        try:
            a = analysis_cuts_dict["lower"].keys()
            for cut_var in analysis_cuts_dict["lower"].keys():
                pass_indices_OS = np.logical_and(pass_indices_OS,
                                                 uncut_data_OS[code][cut_var] >= analysis_cuts_dict["lower"][cut_var])
                pass_indices_SS = np.logical_and(pass_indices_SS,
                                                 uncut_data_SS[code][cut_var] >= analysis_cuts_dict["lower"][cut_var])
        except:
            pass
        try:
            a = analysis_cuts_dict["upper"].keys()
            for cut_var in analysis_cuts_dict["upper"].keys():
                pass_indices_OS = np.logical_and(pass_indices_OS,
                                                 uncut_data_OS[code][cut_var] <= analysis_cuts_dict["upper"][cut_var])
                pass_indices_SS = np.logical_and(pass_indices_SS,
                                                 uncut_data_SS[code][cut_var] <= analysis_cuts_dict["upper"][cut_var])
        except:
            pass

        pass_indices_OS = np.logical_and(pass_indices_OS, (uncut_data_OS[code]["dimu_mass"] >= SB_left) & (
                    uncut_data_OS[code]["dimu_mass"] <= SB_right))
        pass_indices_SS = np.logical_and(pass_indices_SS, (uncut_data_SS[code]["dimu_mass"] >= SB_left) & (
                    uncut_data_SS[code]["dimu_mass"] <= SB_right))

        # apply cuts to oppsign
        for feat in feature_set:
            cut_data_OS[code][feat] = uncut_data_OS[code][feat][pass_indices_OS]
            cut_data_SS[code][feat] = uncut_data_SS[code][feat][pass_indices_SS]

        print(f"{code} OS has shape {cut_data_OS[code][feat].shape} after cuts")
        print(f"{code} SS has shape {cut_data_SS[code][feat].shape} after cuts")


    dataset = {
        "": cut_data_OS,
        "_no_signal": cut_data_SS
    }

    for code in codes_list:
        for postfix, data in dataset.items():
            muon = create_point_cloud(data, code)
            muon["energy"] = muon.E
            # muon = pad_object(muon, 2)
            print(muon)
            dimu_mass = (muon[..., 0] + muon[..., 1]).mass
            HT = (muon[..., 0].pt + muon[..., 1].pt)
            deltaR = muon[..., 0].deltaR(muon[..., 1])
            data_df = dict()

            feature_name = point_cloud_feature_to_save.keys()
            features = []
            for feature in feature_name:
                feature_array = ak.to_numpy(getattr(muon, feature))
                if "log" in point_cloud_feature_to_save[feature]:
                    features.append(np.log1p(np.clip(feature_array, 1e-10, None)))
                else:
                    features.append(feature_array)
            data_df['x'] = np.stack(features, axis=-1)
            data_df['x_mask'] = ak.to_numpy(muon.MASK)

            # met: log_normalize
            # met_phi: normalize
            # nLepton: none
            # nbJet: none
            # nJet: none
            # HT: log_normalize
            # HT_lep: log_normalize
            # M_all: log_normalize
            # M_leps: log_normalize
            # M_bjets: log_normalize
            # deltaR: normalize

            condition_candidates = {
                "met": ak.to_numpy(ak.zeros_like(dimu_mass)),
                "met_phi": ak.to_numpy(ak.zeros_like(dimu_mass)),
                "nLepton": ak.to_numpy(ak.ones_like(dimu_mass) * 2),
                "nbJet": ak.to_numpy(ak.zeros_like(dimu_mass)),
                "nJet": ak.to_numpy(ak.zeros_like(dimu_mass)),
                "HT": ak.to_numpy(HT),
                "HT_lep": ak.to_numpy(HT),
                "inv_mass": ak.to_numpy(dimu_mass),
                "M_leps": ak.to_numpy(dimu_mass),
                "M_bjets": ak.to_numpy(ak.zeros_like(dimu_mass)),
                "deltaR": ak.to_numpy(deltaR)
            }

            condition_array = []
            for condition, cond_norm in condition_feature_to_save.items():
                if "log" in cond_norm:
                    condition_array.append(np.log1p(np.clip(condition_candidates[condition], 1e-10, None)))
                else:
                    condition_array.append(condition_candidates[condition])

            data_df['conditions'] = np.stack(condition_array, axis=-1)

            data_df['conditions_mask'] = np.expand_dims(ak.to_numpy(ak.ones_like(HT, dtype=bool)), axis=1)
            data_df["num_sequential_vectors"] = ak.to_numpy(ak.num(muon))
            data_df["num_vectors"] = data_df["num_sequential_vectors"] + 1
            data_df["assignment_indices"] = data_df['conditions']
            data_df['assignment_indices_mask'] = data_df['conditions']
            data_df["assignment_mask"] = data_df['conditions']
            data_df["classification"] = ak.to_numpy(ak.zeros_like(HT, dtype=int))
            data_df["regression"] = data_df['conditions']
            data_df["regression_mask"] = data_df['conditions']
            data_df["x_invisible"] = np.zeros_like(data_df['x'])
            data_df["x_invisible_mask"] = np.zeros_like(data_df['x_mask'])

            for name, data in data_df.items():
                if isinstance(data, np.ndarray):
                    print(name, np.shape(data))
            mean_x, std_x = mean_std_last_dim(data_df['x'], data_df['x_mask'])
            mean_cond, std_cond = mean_std_last_dim(np.expand_dims(data_df['conditions'], axis=1),
                                                    data_df['conditions_mask'])

            norm_dict = dict()
            norm_dict['input_mean'] = {'Source': torch.tensor(mean_x.flatten(), dtype=torch.float32),
                                       'Conditions': torch.tensor(mean_cond.flatten(), dtype=torch.float32)}
            norm_dict['input_std'] = {'Source': torch.tensor(std_x.flatten(), dtype=torch.float32),
                                      'Conditions': torch.tensor(std_cond.flatten(), dtype=torch.float32)}
            norm_dict['input_num_mean'] = {"Source": torch.tensor([0.0], dtype=torch.float32)}
            norm_dict['input_num_std'] = {"Source": torch.tensor([1.0], dtype=torch.float32)}
            norm_dict['regression_mean'] = {}
            norm_dict['regression_std'] = {}
            norm_dict['class_counts'] = torch.tensor([data_df['x'].shape[0]], dtype=torch.float32)
            norm_dict['class_balance'] = torch.tensor([1], dtype=torch.float32)
            norm_dict['particle_balance'] = {}
            norm_dict['invisible_mean'] = {"Source": norm_dict['input_mean']['Source']}
            norm_dict['invisible_std'] = {"Source": norm_dict['input_std']['Source']}
            norm_dict['subprocess_counts'] = norm_dict['class_counts']
            norm_dict['subprocess_balance'] = norm_dict['class_balance']

            SBL_filter = (dimu_mass < SR_left) & (dimu_mass > SB_left)
            SBR_filter = (dimu_mass < SB_right) & (dimu_mass > SR_right)
            SR_filter = (dimu_mass < SR_right) & (dimu_mass > SR_left)
            SB_filter = SBL_filter | SBR_filter

            storedir = control["output"]["storedir"]
            storedir = clean_and_append(storedir, postfix)
            save_file(
                save_dir=os.path.join(storedir, "SB"),
                data_df=data_df,
                norm_dict=norm_dict,
                event_filter=SB_filter
            )

            save_file(
                save_dir=os.path.join(storedir, "SR"),
                data_df=data_df,
                norm_dict=norm_dict,
                event_filter=SR_filter
            )


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    # Parse command-line arguments
    args = parser.parse_args()

    prepare_CMSOpenData(args)

if __name__ == "__main__":
    main()