import numpy as np
from numpy import ndarray
import vector


def compute_mass(pt, eta, energy):
    pz = pt * np.sinh(eta)
    mass2 = energy ** 2 - (pt ** 2 + (pt * np.sinh(eta)) ** 2)

    mass = np.zeros_like(mass2)
    positive_mask = mass2 > 0
    mass[positive_mask] = np.sqrt(mass2[positive_mask])
    return mass


def get_masses(point_clouds):
    # point_clouds: [batch, n_particles, n_features]
    pt = point_clouds[:, :, 1]  # pt
    eta = point_clouds[:, :, 2]  # eta
    phi = point_clouds[:, :, 3]  # phi
    E   = point_clouds[:, :, 7]  # energy

    btag     = point_clouds[:, :, 4]
    isLepton = point_clouds[:, :, 5]

    vec_all = vector.arr({
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "E": E,
    })

    vec_leps = vector.arr({
        "pt": np.where(isLepton > 0.5, pt, 0.0),
        "eta": np.where(isLepton > 0.5, eta, 0.0),
        "phi": np.where(isLepton > 0.5, phi, 0.0),
        "E": np.where(isLepton > 0.5, E, 0.0),
    })

    vec_bjets = vector.arr({
        "pt": np.where(btag > 0.5, pt, 0.0),
        "eta": np.where(btag > 0.5, eta, 0.0),
        "phi": np.where(btag > 0.5, phi, 0.0),
        "E": np.where(btag > 0.5, E, 0.0)
    })

    return {
        "INPUTS/Conditions/M_all": vec_all.sum(axis=1).mass,
        "INPUTS/Conditions/M_leps": vec_leps.sum(axis=1).mass,
        "INPUTS/Conditions/M_bjets": vec_bjets.sum(axis=1).mass,
    }


def extract_and_split_4vec(pc, indices, base_key):
    """
    Extracts and splits 4-momentum into separate fields (pt, eta, phi, energy).
    """
    feature_idx = [1, 2, 3, 7]  # pt, eta, phi, energy
    feature_names = ['pt', 'eta', 'phi', 'energy']
    n_event = pc.shape[0]

    result = {}
    valid_mask = indices >= 0
    valid_indices = indices[valid_mask]

    for fidx, fname in zip(feature_idx, feature_names):
        output = np.zeros(n_event, dtype=pc.dtype)
        output[valid_mask] = pc[valid_mask, valid_indices, fidx]
        result[f"{base_key}/{fname}"] = output

    return result


def extract_truth_4vec(truth_particles, pdgids, labels, base_prefix):
    """
    Extracts and stores 4-momentum info for given PDGIDs per event.

    truth_particles: structured array of shape (n_event, n_particles)
    pdgids: list of allowed PDGIDs (e.g. [+6, +24, -11, -13])
    labels: list of labels to assign (e.g. ['top', 'W', 'lepton'])
    base_prefix: 'EXTRA/t1' or 'EXTRA/t2'

    Returns: dict with keys like 'EXTRA/t1/top/pt' → shape (n_event,)
    """
    n_event, n_particles = truth_particles.shape
    result = {}
    feature_names = ['pt', 'eta', 'phi', 'mass']

    pdgid_array = truth_particles['PDGID']  # shape: (n_event, n_particles)

    for pdgid_group, label in zip(pdgids, labels):
        mask = np.isin(pdgid_array, pdgid_group)  # shape: (n_event, n_particles)

        # Find first match per event (or -1 if none)
        has_match = mask.any(axis=1)
        first_match_idx = np.where(has_match, mask.argmax(axis=1), -1)  # shape: (n_event,)

        for name in feature_names:
            field = truth_particles[name]  # shape: (n_event, n_particles)
            # Gather values with advanced indexing, safe fallback to 0
            flat_idx = np.arange(n_event)
            values = np.zeros(n_event, dtype=field.dtype)
            valid_mask = first_match_idx != -1
            values[valid_mask] = field[flat_idx[valid_mask], first_match_idx[valid_mask]]

            result[f"{base_prefix}/{label}/{name}"] = values

    return result


def convert_nu2flow(data: dict[str, ndarray]):
    leptons = data['delphes/leptons']
    jets = data['delphes/jets']

    n_event = leptons.shape[0]
    lep_num = 2
    jet_num = data['delphes/jets'].shape[1]
    max_par = 12

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
    point_clouds[:, lep_num:jet_num + lep_num, :] = jet_features

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

    jets = point_clouds[:, lep_num:jet_num + lep_num, :]
    jets_reordered = jets[batch_indices, sorted_indices]
    bjet_idx_reordered = bjet_idx[batch_indices, sorted_indices]

    # Build mask
    cond_first_two = np.all(bjet_idx_reordered[:, :2] == [0, 1], axis=1)
    cond_rest_are_neg1 = np.all(bjet_idx_reordered[:, 2:] == -1, axis=1)
    good_bjet_assign = cond_first_two & cond_rest_are_neg1

    point_clouds[:, lep_num:lep_num + jet_num, :] = jets_reordered

    point_clouds_mask = (point_clouds > 0).sum(axis=-1) > 0

    good_events = (charges.sum(axis=-1) == 0) & good_bjet_assign

    #### For assignments ####
    # lepton charge: plus -> neutrino pdg: plus -> b_jet indices: 0
    # lepton charge: minus -> neutrino pdg: minus -> b_jet indices: 1
    t2_b = np.full((n_event,), -1)
    t2_b = np.where(bjet_idx_reordered[:, 0] == 1, 2, t2_b)
    t2_b = np.where(bjet_idx_reordered[:, 1] == 1, 3, t2_b)

    assignments = {
        'TARGETS/t1/b': np.where(bjet_idx_reordered[:, 0] == 0, 2, -1),
        'TARGETS/t1/l+': np.zeros(n_event, dtype=np.int32),
        'TARGETS/t2/b': t2_b,
        'TARGETS/t2/l-': np.ones(n_event, dtype=np.int32),
    }

    extra = {}
    extra.update(extract_and_split_4vec(point_clouds, assignments['TARGETS/t1/b'], 'EXTRA/t1/b'))
    extra.update(extract_and_split_4vec(point_clouds, assignments['TARGETS/t1/l+'], 'EXTRA/t1/l'))
    extra.update(extract_and_split_4vec(point_clouds, assignments['TARGETS/t2/b'], 'EXTRA/t2/b'))
    extra.update(extract_and_split_4vec(point_clouds, assignments['TARGETS/t2/l-'], 'EXTRA/t2/l'))

    truth_particles = data['delphes/truth_particles']

    # For t1 (positives): top+, W+, lepton+
    t1_info = extract_truth_4vec(
        truth_particles,
        pdgids=[[6], [24], [-11, -13]],
        labels=['t', 'W', 'l'],
        base_prefix='EXTRA/truth_t1'
    )

    # For t2 (negatives): top−, W−, lepton−
    t2_info = extract_truth_4vec(
        truth_particles,
        pdgids=[[-6], [-24], [11, 13]],
        labels=['t', 'W', 'l'],
        base_prefix='EXTRA/truth_t2'
    )
    extra.update(t1_info)
    extra.update(t2_info)

    return {
        'EXTRA/raw_num_bjet': data['delphes/nbjets'],
        'EXTRA/raw_num_jet': data['delphes/njets'],
        'INFO/VetoDoubleAssign': good_events,

        'INPUTS/Source/MASK': point_clouds_mask,
        'INPUTS/Source/energy': point_clouds[:, :, 7],
        'INPUTS/Source/pt': point_clouds[:, :, 1],
        'INPUTS/Source/eta': point_clouds[:, :, 2],
        'INPUTS/Source/phi': point_clouds[:, :, 3],
        'INPUTS/Source/btag': point_clouds[:, :, 4],
        'INPUTS/Source/isLepton': point_clouds[:, :, 5],
        'INPUTS/Source/charge': point_clouds[:, :, 6],

        'INPUTS/Conditions/MASK': np.ones((n_event, 1), dtype=np.float32),
        'INPUTS/Conditions/met': data['delphes/MET']['MET'],
        'INPUTS/Conditions/met_phi': data['delphes/MET']['phi'],
        'INPUTS/Conditions/nLepton': np.ones((n_event,), dtype=np.float32) * lep_num,
        'INPUTS/Conditions/nJet': data['delphes/njets'],
        'INPUTS/Conditions/nbJet': data['delphes/nbjets'],
        'INPUTS/Conditions/HT': point_clouds[:, :, 1].sum(axis=1),
        'INPUTS/Conditions/HT_lep': point_clouds[:, :lep_num, 1].sum(axis=1),
        # 'INPUTS/Conditions/M_all':
        **get_masses(point_clouds),

        'INPUTS/Invisible/MASK': np.ones((n_event, 2), dtype=np.float32),
        'INPUTS/Invisible/pt': data['delphes/neutrinos']['pt'],
        'INPUTS/Invisible/eta': data['delphes/neutrinos']['eta'],
        'INPUTS/Invisible/phi': data['delphes/neutrinos']['phi'],

        **assignments,
        **extra,
    }
