from copy import deepcopy
import re
import awkward as ak
import vector

PDG_Dict = {
    "Z": [23],
    "W": [24, -24],
    "V": [23, 24, -24],
    "l": [11, 13, -11, -13],
    "l+": [11, 13],
    "l-": [-11, -13],
    "t": [6, -6],
    "v": [12, 14, 16, -12, -14, -16],
    "q": [1, 2, 3, 4, -1, -2, -3, -4],
    "b": [5, -5],
    "gamma": [22],
    "H": [25],
    "a": [36],
    "a_twenty": [36],
    "a_thirty": [36],
    "a_forty": [36],
    "a_sixty": [36],
    "W+": [24],
    "W-": [-24],
}

neutrino_representation = []
resonance_representation = []
for pdg_ in PDG_Dict:
    is_full_neutrino = True
    is_resonance_particle = True
    for pdgid in PDG_Dict[pdg_]:
        if not abs(pdgid) in [12, 14, 16]: is_full_neutrino = False
        if not abs(pdgid) in [23, 24, 6]: is_resonance_particle = False
    if is_full_neutrino: neutrino_representation.append(pdg_)
    if is_resonance_particle: resonance_representation.append(pdg_)


def return_last_product(process_diagram, last_product=None):
    if last_product is None:
        last_product = dict()
    for product in process_diagram:
        if product == 'SYMMETRY': continue
        if process_diagram[product] is None:
            product_type = re.sub(r'\d+', '', product)
            if product_type not in last_product:
                last_product[product_type] = 1
            else:
                last_product[product_type] += 1
        else:
            return_last_product(process_diagram[product], last_product)
    return last_product


def select_by_pdgId(candidate, candidate_name):
    candidate_type = re.sub(r'\d+', '', candidate_name)
    candidate_pdgId = PDG_Dict[candidate_type]
    select = ak.values_astype(ak.zeros_like(candidate.pdgId), bool)
    for pdgId in candidate_pdgId:
        select = select | (candidate.pdgId == pdgId)
    return candidate[select]


def select_by_mother_status(parton, candidate, mother_status):
    cartesian = ak.argcartesian([parton.index, candidate.M1], axis=1)
    matches = cartesian[(parton.index[cartesian["0"]] == candidate.M1[cartesian["1"]])]

    mother = parton[matches["0"]]
    candidate = candidate[matches["1"]]

    return candidate[mother.status == mother_status]


def select_by_rank(Daughter, Mother, rank=0):
    order = ak.argsort(Mother.index)
    Daughter = Daughter[order]
    Mother = Mother[order]
    non_empty_mask = ak.num(Mother) > 0  # Check which sublists are non-empty
    run_length = ak.flatten((ak.run_lengths(Mother[non_empty_mask].index)))  # Filter out empty sublists

    Daughter = ak.unflatten(Daughter, counts=run_length, axis=1)
    Mother = ak.unflatten(Mother, counts=run_length, axis=1)
    Daughter = ak.pad_none(Daughter, rank + 1, axis=-1)
    Mother = ak.pad_none(Mother, rank + 1, axis=-1)
    Daughter = ak.flatten(ak.drop_none(ak.from_regular(Daughter[..., [rank]]), axis=1), axis=-1)
    Mother = ak.flatten(ak.drop_none(ak.from_regular(Mother[..., [rank]]), axis=1), axis=-1)
    return Daughter, Mother


def select_by_products(parton, candidate_array, products, candidate_name, process_summary=None, rank=None):
    if process_summary is None:
        process_summary = dict()
    if products is None:
        return candidate_array, process_summary

    if products is not None and "SYMMETRY" in products:
        symmetry = products["SYMMETRY"]
        symmetry_map = {sym_: idx for idx, sym_ in enumerate(symmetry)}
    else:
        symmetry_map = None

    for product_idx, product in enumerate(products):
        if (product == "SYMMETRY"): continue
        product_name = '{}/{}'.format(candidate_name, product)
        product_array = select_by_pdgId(parton, product)
        cartesian = ak.argcartesian([product_array.M1, candidate_array.index], axis=1)
        matches = cartesian[(product_array.M1[cartesian["0"]] == candidate_array.index[cartesian["1"]])]

        product_from_mother = product_array[matches["0"]]
        candidate_array = candidate_array[matches["1"]]

        candidate_array[product_name] = product_from_mother.index

        # print('---product decay---')
        # for i in product_from_mother[0]:
        #     print(candidate_name, i)

        if products[product] is not None:

            ranking = None if ((symmetry_map is None) or (product not in symmetry_map)) else symmetry_map[product]
            product_decay_array, process_summary = select_by_products(
                parton, product_from_mother, products[product], product_name, rank=ranking
            )
            # print('---product decay---')
            # for i in product_decay_array[1]:
            #     print(candidate_name, i)
            cartesian_selection = ak.argcartesian([product_decay_array.M1, candidate_array.index], axis=1)
            matches_selection = cartesian_selection[
                (product_decay_array.M1[cartesian_selection["0"]] == candidate_array.index[cartesian_selection["1"]])]
            product_from_mother = product_decay_array[matches_selection["0"]]
            candidate_array = candidate_array[matches_selection["1"]]
            for sub_product in product_from_mother.fields:
                if sub_product in candidate_array.fields: continue
                candidate_array[sub_product] = product_from_mother[sub_product]

        if symmetry_map is not None and product in symmetry_map:
            product_from_mother, candidate_array = select_by_rank(
                product_from_mother, candidate_array, symmetry_map[product]
            )

        if rank is not None and product_idx == 0:
            candidate_array = ak.pad_none(candidate_array, rank + 1, axis=1)
            candidate_array = ak.from_regular(candidate_array[..., [rank]])
            candidate_array = ak.drop_none(candidate_array, axis=1)

        process_summary[product_name] = candidate_array[product_name]

    return candidate_array, process_summary


def assignment(candidate_array, products, candidate_name, process_summary):
    if products is None:
        return process_summary

    for product in products:
        if product == "SYMMETRY": continue
        product_name = '{}/{}'.format(candidate_name, product)
        process_summary[product_name] = candidate_array[product_name]
        if products[product] is not None:
            process_summary = assignment(candidate_array, products[product], product_name, process_summary)
    return process_summary


def assign_Reco_LorentzVector(Event_dict, products, parton, candidate_name, reconstructed_momentum_dict):
    default_reco_v4 = vector.zip({
        "pt": -1,
        "eta": -1,
        "phi": -1,
        "mass": -1,
    })

    if products is None:
        return reconstructed_momentum_dict

    for product in products:
        if product == 'SYMMETRY': continue
        product_name = '{}/{}'.format(candidate_name, product)
        if products[product] is not None:
            reconstructed_momentum_dict = assign_Reco_LorentzVector(Event_dict, products[product], parton, product_name,
                                                                    reconstructed_momentum_dict)
        else:
            reconstructed_momentum_dict[product_name] = ak.fill_none(
                ak.pad_none(
                    parton[ak.unflatten(Event_dict[product_name], counts=1, axis=-1) == parton.index].reco_v4,
                    1, axis=-1
                )[..., 0],
                default_reco_v4
            )  # TODO: There seems to have parton sharing same index. Probably saving dulplicate parton. Not leading problem to training, but may need to take care.
            # print(product_name, ak.type(reconstructed_momentum_dict[product_name]))

        if candidate_name not in reconstructed_momentum_dict:
            reconstructed_momentum_dict[candidate_name] = reconstructed_momentum_dict[product_name]
        #      print(candidate_name, 'first assign', ak.type(ak.flatten(Reconstructed_momentum_dict[candidate_name])))
        else:
            #      print(candidate_name, 'second assign', ak.type(ak.flatten(Reconstructed_momentum_dict[product_name])))
            reconstructed_momentum_dict[candidate_name] = ak.where(
                (reconstructed_momentum_dict[product_name].pt > 0) & (
                        reconstructed_momentum_dict[candidate_name].pt > 0),
                (reconstructed_momentum_dict[candidate_name].add(reconstructed_momentum_dict[product_name])),
                (ak.full_like(reconstructed_momentum_dict[product_name], -1.0)))

    return reconstructed_momentum_dict


def assign_matched_index(Event_dict, products, parton, candidate_name, matched_index_dict=None):
    if products is None:
        return matched_index_dict

    for product in products:
        if product == 'SYMMETRY': continue
        product_name = '{}/{}'.format(candidate_name, product)
        if products[product] is not None:
            matched_index_dict = assign_matched_index(
                Event_dict, products[product], parton, product_name, matched_index_dict
            )
        else:
            matched_index_dict[product_name] = ak.fill_none(ak.pad_none(
                parton[parton.index == ak.unflatten(Event_dict[product_name], counts=1, axis=-1)]["matched_index"], 1,
                axis=-1)[..., 0], -1)
            matched_index_dict[product_name] = ak.unflatten(matched_index_dict[product_name], counts=1, axis=-1)
            # print('product type', ak.type(matched_index_dict[product_name]))
    return matched_index_dict
