import os
import argparse
import h5py
import glob
import numpy as np
import vector
from preprocessing.dqm_util import get_hist, draw_hist, find_dataset_name, valid_components_mask
from tqdm import tqdm

vector.register_awkward()
from preprocessing.process_info import *
import re


def pad_object(obj, nMax):
    pad_awkward = ak.pad_none(obj, target=nMax, clip=True)
    pad_numpy = ak.to_numpy(pad_awkward)  # pad_numpy.data none entry default to be 0.0
    pad_numpy.data[pad_numpy.mask] = 0.
    return pad_numpy.data


def select_event(obj):
    nObj = ak.num(obj["els"]) + ak.num(obj["mus"]) + ak.num(obj["jets"])
    for key in obj:
        obj[key] = obj[key][nObj > 1]
    return obj


def lorentz_vector(obj, Type='Jet', pad_length=None):
    v4 = vector.zip(
        {
            "pt": obj[:, :, 0],
            "eta": obj[:, :, 1],
            "phi": obj[:, :, 2],
            "mass": obj[:, :, 3],
        }
    )

    if pad_length is not None:
        v4 = ak.pad_none(v4, target=pad_length, axis=1, clip=True)

    if (Type == 'els') or (Type == 'mus'):
        v4["isLepton"] = ak.values_astype(ak.ones_like(v4.pt), bool)
        v4["isBTag"] = ak.values_astype(ak.zeros_like(v4.pt), bool)
        v4["charge"] = obj[:, :, 4]
    elif Type == 'jets':
        v4["isLepton"] = ak.values_astype(ak.zeros_like(v4.pt), bool)
        v4["isBTag"] = obj[:, :, 4]
        v4["charge"] = ak.values_astype(ak.zeros_like(v4.pt), bool)
    else:
        v4["isLepton"] = ak.values_astype(ak.zeros_like(v4.pt), bool)
        v4["isBTag"] = ak.values_astype(ak.zeros_like(v4.pt), bool)
        v4["charge"] = ak.values_astype(ak.zeros_like(v4.pt), bool)

    return v4


def build_output_name(product: str, process: str, diagram: dict, tag: str = "TARGETS") -> str:
    parts = product.replace("EVENT", tag).split("/")
    if len(parts) > 2:
        if diagram.get("store_full_name", False):
            return "/".join([parts[0], process] + parts[1:])
        else:
            return "/".join([parts[0], process, parts[1], parts[-1]])
    else:
        return "/".join([parts[0], process, parts[1]])


def store_valid_components(data_dict, base_name, momentum, mask, components):
    for comp in components:
        arr = getattr(momentum, comp)
        data_dict[f"{base_name}/{comp}"] = np.nan_to_num(ak.to_numpy(arr), nan=0.0)
    data_dict[f"{base_name}/MASK"] = valid_components_mask(momentum, mask, components)


def build_dataset_with_matching(objects, diagram, process, dqm_plot: dict, return_data: bool = True):
    v4 = dict()
    for obj in ["els", "jets", "mus", "tas", "genpart"]:
        v4[obj] = lorentz_vector(objects[obj], obj)

    n_lepton = ak.num(objects["els"]) + ak.num(objects["mus"])

    n_gen_lepton = 0
    last_product = return_last_product(diagram['diagram'])
    for product_ in last_product:
        if "l" in product_:
            n_gen_lepton += last_product[product_]
    # print("nGenLepton", n_gen_lepton)

    # ---------------------------
    # Reconstructed:
    #   -1: initial state, don't need reconstruction
    #    0: mediate state, need reconstruction
    #    1: final state, already reconstructed
    # ----------------------------

    reco_v4 = vector.zip(
        {
            "pt": ak.ones_like(objects["genpart"][:, :, 8]) * -1,
            "eta": ak.ones_like(objects["genpart"][:, :, 8]) * -1,
            "phi": ak.ones_like(objects["genpart"][:, :, 8]) * -1,
            "mass": ak.ones_like(objects["genpart"][:, :, 8]) * -1
        }
    )

    v4_combined = ak.concatenate((v4["jets"], v4["els"], v4["mus"], v4["tas"]), axis=1)

    pdg_id = ak.values_astype(objects["genpart"][:, :, 7], int)

    matched_index = ak.values_astype(objects["genpart"][:, :, 9], int)
    matched_index = ak.where(
        ((matched_index > -1) & (abs(pdg_id) == 11)),
        ak.unflatten(ak.num(v4["jets"]), counts=1, axis=0) + matched_index,
        matched_index
    )
    matched_index = ak.where(
        ((matched_index > -1) & (abs(pdg_id) == 13)),
        ak.unflatten(ak.num(v4["jets"]) + ak.num(v4["els"]), counts=1, axis=0) + matched_index,
        matched_index
    )
    matched_index = ak.where(
        ((matched_index > -1) & (abs(pdg_id) == 15)),
        ak.unflatten(ak.num(v4["jets"]) + ak.num(v4["els"]) + ak.num(v4["mus"]), counts=1, axis=0) + matched_index,
        matched_index
    )

    matched_index_safe = ak.where((matched_index > -1), matched_index, 0)  # The padded one will not be used in any case

    # print(ak.num(v4_combined))
    reco_v4 = ak.where((matched_index > -1), v4_combined[matched_index_safe], reco_v4)
    reco_v4 = ak.where(
        (abs(pdg_id) == 12) | (abs(pdg_id) == 14) | (abs(pdg_id) == 16),
        ak.from_regular(v4["genpart"]),
        reco_v4
    )

    parton = ak.zip(
        {
            "index": objects["genpart"][:, :, 4],
            "pdgId": objects["genpart"][:, :, 7],
            "M1": objects["genpart"][:, :, 5],
            "status": objects["genpart"][:, :, 8]
        }
    )

    # for i in range(2):
    #     for j in range(len(parton[i])):
    #         print("events: ", i, "partons: ", j, parton[i][j])

    event_dict = dict()
    candidate_dict = dict()
    for product in diagram['diagram']:
        if product == "SYMMETRY": continue
        if "SYMMETRY" in diagram['diagram']:
            symmetry = diagram['diagram']["SYMMETRY"]
            symmetry_map = {sym_: idx for idx, sym_ in enumerate(symmetry)}
        else:
            symmetry_map = None

        candidate_array = select_by_pdgId(parton, product)
        candidate_array = select_by_mother_status(parton, candidate_array, 21)
        product_name = 'EVENT/{}'.format(product)
        ranking = None if ((symmetry_map is None) or (product not in symmetry_map)) else symmetry_map[product]
        candidate_array, event_dict = select_by_products(
            parton, candidate_array, diagram['diagram'][product],
            product_name, event_dict, rank=ranking
        )
        candidate_dict[product] = candidate_array

    event_dict = dict()
    for product in diagram['diagram']:
        if product == 'SYMMETRY': continue
        product_name = 'EVENT/{}'.format(product)
        event_dict[product_name] = candidate_dict[product].index
        event_dict = assignment(candidate_dict[product], diagram['diagram'][product], product_name, event_dict)
    for ele_idx in event_dict:
        dqm_plot['nFoundParton_{}'.format(ele_idx).replace('/', '_')] = get_hist(
            [2, -0.5, 1.5], ak.num(event_dict[ele_idx])
        )
        event_dict[ele_idx] = ak.flatten(
            ak.fill_none(ak.pad_none(event_dict[ele_idx], 1, axis=1), -1, axis=1), axis=1
        )
        # print('Event', ele_idx, ak.type(event_dict[ele_idx]), event_dict[ele_idx])

    gen_v4 = vector.zip(
        {
            "pt": objects["genpart"][:, :, 0],
            "eta": objects["genpart"][:, :, 1],
            "phi": objects["genpart"][:, :, 2],
            "mass": objects["genpart"][:, :, 3]
        }
    )
    parton["gen_v4"] = gen_v4
    parton["reco_v4"] = reco_v4
    parton["matched_index"] = matched_index

    reconstructed_momentum = dict()
    matched_index_dict = dict()
    gen_level_momentum = dict()
    for product in diagram['diagram']:
        if product == 'SYMMETRY': continue
        product_name = 'EVENT/{}'.format(product)
        reco_dict_ = assign_Reco_LorentzVector(
            event_dict, diagram['diagram'][product], parton, product_name, reconstructed_momentum
        )
        matched_index_dict = assign_matched_index(
            event_dict, diagram['diagram'][product], parton, product_name, matched_index_dict
        )

        # for reco_ in matched_index_dict:
        #     print('index', reco_, matched_index_dict[reco_])

        for reco_ in reco_dict_:
            reconstructed_momentum[reco_] = reco_dict_[reco_]
            # print('recopt', reco_, reconstructed_momentum[reco_].pt)

    default_reco_v4 = vector.zip({
        "pt": -1,
        "eta": -1,
        "phi": -1,
        "mass": -1,
    })
    for particle_ in event_dict:
        gen_level_momentum[particle_] = ak.fill_none(
            ak.pad_none(
                parton[ak.unflatten(event_dict[particle_], counts=1, axis=-1) == parton.index].gen_v4, 1,
                axis=-1
            )[..., 0], default_reco_v4
        )
        # print("gen level", particle_, gen_level_momentum[particle_].pt)

    combined_index = None
    for product in matched_index_dict:
        if combined_index is None:
            combined_index = matched_index_dict[product]
        else:
            combined_index = ak.concatenate((combined_index, matched_index_dict[product]), axis=1)
            # DQM_Plot['{}_nMatched'.format(product.replace('/','_'))] =  Get_hist([13, -0.5, 12.5], ak.num(Matched_Index_Dict[product][Matched_Index_Dict[product] > -1]))

        matched_index_dict[product] = ak.flatten(
            ak.fill_none(ak.pad_none(matched_index_dict[product], 1, axis=1), -1, axis=1), axis=1)

    nMaxJet = 18
    nevents = len(v4_combined.pt)

    if combined_index is not None:
        combined_index = ak.from_regular(ak.fill_none(combined_index, -1))
        Sorted_index = ak.sort(combined_index, axis=-1)
        Sorted_index = Sorted_index[Sorted_index > -1]
        run_lengths = ak.run_lengths(Sorted_index)
        unique_check = ak.all(run_lengths == 1, axis=-1)
        dqm_plot['GenPart_matching_doubleAssignment'] = get_hist([2, -0.5, 1.5], ak.values_astype(unique_check, int))
        dqm_plot['GenPart_matching_numAssignment'] = get_hist(
            [13, -0.5, 12.5],
            ak.num(combined_index[combined_index > -1])
        )

    else:
        # print("Assign no veto double assignment")
        unique_check = ak.values_astype(ak.ones_like(objects['event'][:, 0]), bool)

    if return_data:
        data_dict = dict()
        mask = ak.to_numpy(ak.pad_none(v4_combined.pt, target=nMaxJet, clip=True)).mask
        data_dict['INPUTS/Source/MASK'] = ~mask
        data_dict['INPUTS/Source/pt'] = pad_object(v4_combined.pt, nMaxJet)
        data_dict['INPUTS/Source/eta'] = pad_object(v4_combined.eta, nMaxJet)
        data_dict['INPUTS/Source/phi'] = pad_object(v4_combined.phi, nMaxJet)
        data_dict['INPUTS/Source/mass'] = pad_object(v4_combined.mass, nMaxJet)
        data_dict['INPUTS/Source/btag'] = pad_object(v4_combined.isBTag, nMaxJet)
        data_dict['INPUTS/Source/isLepton'] = pad_object(v4_combined.isLepton, nMaxJet)
        data_dict['INPUTS/Source/charge'] = pad_object(v4_combined.charge, nMaxJet)
        data_dict['INPUTS/Conditions/met'] = ak.to_numpy(objects['event'][:, 0])
        data_dict['INPUTS/Conditions/met_phi'] = ak.to_numpy(objects['event'][:, 1])
        # Store matched indices
        for product, value in matched_index_dict.items():
            output_name = build_output_name(product, process, diagram, tag="TARGETS")
            data_dict[output_name] = ak.to_numpy(value)

        # Store gen-level momenta
        for product, momentum in gen_level_momentum.items():
            last_particle = re.sub(r"\d+", "", product.split("/")[-1])
            if last_particle not in (neutrino_representation + resonance_representation):
                continue

            output_name = build_output_name(product, process, diagram, tag="REGRESSIONS")

            base_mask = (n_lepton == n_gen_lepton) & (momentum.pt > 0)
            data_dict[f"{output_name}/MASK"] = ak.to_numpy(base_mask)

            if last_particle in neutrino_representation:
                store_valid_components(
                    data_dict, output_name, momentum, data_dict[f"{output_name}/MASK"],
                    ("px", "py", "pz")
                )
            elif last_particle in resonance_representation:
                store_valid_components(
                    data_dict, output_name, momentum, data_dict[f"{output_name}/MASK"],
                    ("pt", "eta", "phi")
                )

        data_dict['INFO/VetoDoubleAssign'] = ak.to_numpy(unique_check)
        if combined_index is not None:
            data_dict['INFO/numAssign'] = ak.to_numpy(ak.num(combined_index[combined_index > -1]))
        else:
            data_dict['INFO/numAssign'] = ak.to_numpy(ak.zeros_like(objects['event'][:, 0]))

        return reconstructed_momentum, dqm_plot, data_dict

    return reconstructed_momentum, dqm_plot, None


def monitor_gen_matching(in_dir, process, feynman_diagram_process, out_dir=None, monitor_plots: bool = False):
    ################################
    # Collect all necessary arrays #
    ################################

    data_dict = dict()
    dataset_structure = None

    process_files = [
        h5name for h5name in glob.glob(os.path.join(in_dir, '{process}_*.h5'.format(process=process)))
        if re.match(os.path.join(in_dir, "{process}_[0-9]+\\.h5".format(process=process)), h5name)
    ]

    if len(process_files) == 0:
        print(f"[Warning] No files found for process: {process}")
        return None

    for h5name in tqdm(process_files, desc=f'{process} -- Loading files', unit='file'):
        # print(h5name)
        h5fr = h5py.File(h5name, mode='r')

        if dataset_structure is None:
            dataset_structure = find_dataset_name(h5fr, list(h5fr))
            # print(dataset_structure)
            for key_ in dataset_structure:
                data_dict[key_] = np.array(list(h5fr[key_]))

        else:
            for key_ in dataset_structure:
                data_dict[key_] = np.concatenate([data_dict[key_], np.array(list(h5fr[key_]))])

        h5fr.close()

        # break # TODO remove this break to process all files


    ##########################
    # Plot Basic Information #
    ##########################

    # Convert numpy to awkward array
    objects = dict()
    for key_ in data_dict:
        if key_ == 'event':
            object_ = ak.flatten((ak.from_numpy(data_dict[key_])), axis=1)
            objects[key_] = object_
            continue
        object_ = ak.from_regular(ak.from_numpy(data_dict[key_]))
        if key_ == 'genpart':
            object_ = object_[object_[:, :, 8] > 0]  # GenParton select with status
        else:
            object_ = object_[object_[:, :, 0] > 0]
        objects[key_] = object_

    objects = select_event(objects)
    # for key_ in objects:
    #     print(key_, ak.type(objects[key_]))
    del data_dict

    dqm_plot = dict()

    dqm_plot['nLepton'] = get_hist([6, -0.5, 5.5], ak.num(objects["els"]) + ak.num(objects["mus"]))
    dqm_plot['nJet'] = get_hist([6, -0.5, 5.5], ak.num(objects["jets"]))
    dqm_plot['nPart_final_product'] = get_hist(
        [9, -0.5, 8.5],
        ak.num(objects["genpart"][objects["genpart"][:, :, 8] == 23])
    )
    dqm_plot['nPart_matched'] = get_hist([6, -0.5, 5.5], ak.num(objects["genpart"][objects["genpart"][:, :, 9] > -1]))
    dqm_plot['GentPart_Flavour'] = get_hist([49, -24.5, 24.5], objects["genpart"][:, :, 7])

    ###################
    ## Gen Matching  ##
    ###################
    reco, dqm_plot, processed_data = build_dataset_with_matching(
        objects, feynman_diagram_process,
        process,
        dqm_plot,
        return_data=True,
    )

    if monitor_plots:
        plot_dir = os.path.join(out_dir, process)
        for reco_name in reco:
            reco_v4 = reco[reco_name]
            dqm_plot[reco_name.replace('/', '_')] = get_hist([300, 0, 300], reco_v4[(reco_v4.rho > 0)].tau)
        sorted_index = ak.sort(ak.values_astype(objects["genpart"][:, :, 9], int), axis=-1)
        sorted_index = sorted_index[sorted_index > -1]
        run_lengths = ak.run_lengths(sorted_index)
        unique_check = ak.values_astype(ak.all(run_lengths == 1, axis=-1), int)
        dqm_plot['GenPart_doubleAssignment'] = get_hist([2, -0.5, 1.5], unique_check)

        for histogram_name in dqm_plot:
            draw_hist(dqm_plot[histogram_name], histogram_name, plot_dir, histogram_name)

    return processed_data


if __name__ == '__main__':

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str, default='DQM_Plot')
    parser.add_argument('--store_dir', type=str, default='Storage')
    parser.add_argument('--process', type=str)
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--store', action='store_true')
    config = parser.parse_args()
    if config.scan:
        # for process_ in Feynman_diagram:
        for process_ in []:
            store_string = '--store' if config.store else ''
            os.system(
                f'python3 Monitor_GenMatching.py --indir {config.indir} --outdir {config.outdir} --store_dir {config.store_dir} --process {process_} {store_string}')
    else:
        monitor_gen_matching(
            config.indir,
            config.process,
            # config.outdir,
            monitor_plots=False
        )
