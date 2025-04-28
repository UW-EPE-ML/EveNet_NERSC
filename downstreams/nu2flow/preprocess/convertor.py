import numpy as np
from numpy import ndarray
import vector


def compute_mass(pt, eta, energy):
    pz = pt * np.sinh(eta)
    mass2 = energy ** 2 - (pt ** 2 + pz ** 2)
    mass = np.where(mass2 > 0, np.sqrt(mass2), 0.0)
    return mass


def convert_nu2flow(data: dict[str, ndarray]):
    leptons = data['delphes/leptons']
    jets = data['delphes/jets']

    n_event = leptons.shape[0]
    lep_num = 2
    jet_num = 6
    max_par = lep_num + jet_num

    point_clouds = np.zeros((n_event, max_par, 8), dtype=np.float32)

    # first jet, then leptons
    # feature: mass, pt, eta, phi, is_bjet, is_lepton, charge, energy

    # Leptons
    pt_l, eta_l, phi_l, energy_l, charge_l = leptons['pt'], leptons['eta'], leptons['phi'], leptons['energy'], leptons[
        'charge']
    mass_l = compute_mass(pt_l, eta_l, energy_l)

    lepton_features = np.stack([
        mass_l, pt_l, eta_l, phi_l,
        np.zeros_like(pt_l),
        np.ones_like(pt_l),
        charge_l.astype(np.float32),
        energy_l
    ], axis=-1)

    # Jets
    pt_j, eta_j, phi_j, energy_j, is_tagged_j = jets['pt'], jets['eta'], jets['phi'], jets['energy'], jets['is_tagged']
    mass_j = compute_mass(pt_j, eta_j, energy_j)

    jet_features = np.stack([
        mass_j, pt_j, eta_j, phi_j,
        is_tagged_j.astype(np.float32),
        np.zeros_like(pt_j),
        np.zeros_like(pt_j),
        energy_j
    ], axis=-1)

    # Fill
    point_clouds[:, :lep_num, :] = lepton_features
    point_clouds[:, lep_num:max_par, :] = jet_features

    # Sawp leptons (plus, minus)
    charges = point_clouds[:, :2, 6]  # shape (n_event, 2)
    need_swap = charges[:, 0] < 0  # (n_event,) boolean
    point_clouds[need_swap, 0, :], point_clouds[need_swap, 1, :] = \
        point_clouds[need_swap, 1, :], point_clouds[need_swap, 0, :]

    # Swap b-jets (0 = b-jet, 1 = anti b-jet)
    # Extract jets
    bjet_idx = data['delphes/jets_indices']  # (n_event, 6)
    # bjet_idx: (n_event, 6), values 0 (b), 1 (anti-b), -1 (others)

    # Map b=0 -> 0, anti-b=1 -> 1, others=-1 -> 2
    sort_key = np.where(bjet_idx == -1, 2, bjet_idx)
    sorted_indices = np.argsort(sort_key, axis=1)
    batch_indices = np.arange(bjet_idx.shape[0])[:, None]  # (n_event, 1)

    jets = point_clouds[:, 2:, :]
    jets_reordered = jets[batch_indices, sorted_indices]
    bjet_idx_reordered = bjet_idx[batch_indices, sorted_indices]

    point_clouds[:, 2:, :] = jets_reordered

    point_clouds_mask = (point_clouds > 0).sum(axis=-1) > 0

    good_events = (charges.sum(axis=-1) == 0)

    #### For assignments ####
    # lepton charge: plus -> neutrino pdg: plus -> b_jet indices: 0
    # lepton charge: minus -> neutrino pdg: minus -> b_jet indices: 1
    t2_b = np.full((n_event,), -1)
    t2_b = np.where(bjet_idx_reordered[:, 0] == 1, 2, t2_b)
    t2_b = np.where(bjet_idx_reordered[:, 1] == 1, 3, t2_b)

    assignments = {
        'TARGETS/t1/b': np.where(bjet_idx_reordered[:,0] == 0, 2, -1),
        'TARGETS/t1/l+': np.zeros(n_event, dtype=np.int32),
        'TARGETS/t2/b': t2_b,
        'TARGETS/t2/l-': np.ones(n_event, dtype=np.int32),
    }


    return {
        'INFO/VetoDoubleAssign': good_events,

        'INPUTS/Source/MASK': point_clouds_mask,
        'INPUTS/Source/mass': point_clouds[:, :, 0],
        'INPUTS/Source/pt': point_clouds[:, :, 1],
        'INPUTS/Source/eta': point_clouds[:, :, 2],
        'INPUTS/Source/phi': point_clouds[:, :, 3],
        'INPUTS/Source/btag': point_clouds[:, :, 4],
        'INPUTS/Source/isLepton': point_clouds[:, :, 5],
        'INPUTS/Source/charge': point_clouds[:, :, 6],

        'INPUTS/Conditions/MASK': np.ones((n_event, 1), dtype=np.float32),
        'INPUTS/Conditions/met': data['delphes/MET']['MET'],
        'INPUTS/Conditions/met_phi': data['delphes/MET']['phi'],

        'INPUTS/Invisible/MASK': np.ones((n_event, 2), dtype=np.float32),
        'INPUTS/Invisible/mass': np.zeros((n_event, 2), dtype=np.float32),
        'INPUTS/Invisible/pt': data['delphes/neutrinos']['pt'],
        'INPUTS/Invisible/eta': data['delphes/neutrinos']['eta'],
        'INPUTS/Invisible/phi': data['delphes/neutrinos']['phi'],
        'INPUTS/Invisible/btag': np.zeros((n_event, 2), dtype=np.float32),
        'INPUTS/Invisible/isLepton': np.zeros((n_event, 2), dtype=np.float32),
        'INPUTS/Invisible/charge': np.zeros((n_event, 2), dtype=np.float32),

        **assignments,
    }
