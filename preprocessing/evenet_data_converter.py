from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from typing import Union, Tuple, List, Optional, Dict
from collections import OrderedDict
from evenet.control.event_info import EventInfo
import awkward as ak


class EveNetDataConverter:
    def __init__(
            self,
            raw_data: dict,
            event_info: EventInfo,
            process: str,
    ):

        self.raw_data = raw_data
        self.event_info = event_info
        self.process = process
        self.total_length = len(self.raw_data["INFO/VetoDoubleAssign"])

    def load_sources(self):
        label = "INPUTS"
        output_dict = {}
        i = 0
        for folder_key, features in self.event_info.input_features.items():
            # print(folder_key)
            features_ak = []
            for feature_info in features:
                # print(feature_info)
                k = f"{label}/{folder_key}/{feature_info[0]}"
                log_scale = feature_info[2]
                # print(k, log_scale)
                if log_scale:
                    features_ak.append(np.log(self.raw_data[k] + 1))
                else:
                    features_ak.append(self.raw_data[k])
            features_ak = np.stack(features_ak, axis=-1)  # (num_events, num_features)
            mask_np = ~np.all(features_ak == 0, axis=-1)

            output_dict[f"sources-{i}-data"] = features_ak
            output_dict[f"sources-{i}-mask"] = mask_np
            i += 1
        return output_dict

    def load_assignments(self, assignment_names, assignment_map, direct_from_spanet=False):
        label = "TARGETS"
        output_dict = OrderedDict()

        n_targets = len(assignment_names)
        max_daughters = max(len(daughters) for _, _, daughters in assignment_map)
        num_events = self.total_length

        # Init arrays
        full_indices = np.full((num_events, n_targets, max_daughters), -1, dtype=int)
        full_mask = np.zeros((num_events, n_targets), dtype=bool)
        index_mask = np.zeros((num_events, n_targets, max_daughters), dtype=bool)

        for row_idx, (process, product, daughters) in enumerate(assignment_map):
            if process != self.process:
                continue  # Skip if this target doesn't belong to current process

            if not daughters:
                print(f"[WARN] No daughters for {process}/{product}, skipping.")
                continue

            target_prefix = f"{label}/{process}/{product}" if not direct_from_spanet else f"{label}/{product}"
            daughter_fields = [f"{target_prefix}/{d}" for d in daughters]

            try:
                indices = np.stack([self.raw_data[field] for field in daughter_fields], axis=-1)
            except KeyError as e:
                print(f"[WARN] Missing field {e}; skipping {target_prefix}")
                continue

            mask = np.all(indices >= 0, axis=1)

            full_indices[:, row_idx, :len(daughters)] = indices
            full_mask[:, row_idx] = mask
            index_mask[:, row_idx, :len(daughters)] = True

            print(f"[INFO] ASSIGNMENT Loaded {target_prefix}")

        output_dict["assignments-indices"] = full_indices
        output_dict["assignments-mask"] = full_mask
        output_dict["assignments-indices-mask"] = index_mask

        return output_dict

    def load_regressions(self, regression_keys, regression_key_map, direct_from_spanet=False):
        label = "REGRESSIONS"
        output_dict = OrderedDict()

        num_events = self.total_length

        num_targets = len(regression_keys)
        regression_data = np.zeros((num_events, num_targets), dtype=np.float32)
        regression_mask = np.zeros((num_events, num_targets), dtype=bool)

        # Fill in data and mask
        for idx, (process, particle, product, target_name) in enumerate(regression_key_map):
            if direct_from_spanet:
                data_key = f"{label}/{process}/{particle}"
                mask_key = f"{label}/{process}/{particle}"
                regression_data[:, idx] = self.raw_data[data_key]
                regression_mask[:, idx] = np.ones_like(regression_data[:, idx])
            else:
                key = f"{process}/{particle}"
                if product is not None:
                    data_key = f"{label}/{key}/{product}/{target_name}"
                else:
                    data_key = f"{label}/{key}/{target_name}"
                mask_key = f"{label}/{key}/MASK"

                try:
                    regression_data[:, idx] = self.raw_data[data_key]
                    regression_mask[:, idx] = self.raw_data[mask_key]
                except KeyError as e:
                    # print(f"[WARN] Missing {data_key} or {mask_key}, skipping: {e}")
                    continue
            print(f"[INFO] Recorded {data_key} and {mask_key}")

        # Store in output
        output_dict["regression-data"] = regression_data
        output_dict["regression-mask"] = regression_mask

        return output_dict

    def load_classification(self):
        output_dict = OrderedDict()
        label = "CLASSIFICATIONS"

        output_dict[f"classification-EVENT/signal"] = self.raw_data[f"{label}/EVENT/signal"]
        return output_dict

    def filter_process(self, process: str, process_info: dict):
        process_id = process_info['process_id']
        category = process_info['category']

        selection = self.raw_data["INFO/VetoDoubleAssign"]
        data_selected = {key: arr[selection] for key, arr in self.raw_data.items()}

        data_selected['CLASSIFICATIONS/EVENT/signal'] = np.zeros_like(
            data_selected['INFO/VetoDoubleAssign'], dtype=int
        ) + process_id

        if process in self.event_info.event_mapping:
            subprocess_id = list(self.event_info.event_mapping.keys()).index(process)
        else:
            subprocess_id = -1
        data_selected['METADATA/PROCESS'] = np.zeros_like(
            data_selected['INFO/VetoDoubleAssign'], dtype=int
        ) + subprocess_id

        n_event_original = len(selection)
        n_event_current = len(data_selected['INFO/VetoDoubleAssign'])

        print(f"Veto Double Assignment:  {n_event_current}/{n_event_original}. [{category}: {self.process}]")

        self.raw_data = data_selected
        self.total_length = n_event_current

    def load_invisible(self, max_num_neutrinos: int, direct_from_spanet: bool = False):
        source_len = len(self.event_info.input_features['Source'])
        feature_len = len(self.event_info.generations['Neutrinos'])

        assert source_len == feature_len, "Mismatch in feature length, check Generation[Neutrinos] block in event_info"

        x_inv = np.zeros((self.total_length, max_num_neutrinos, feature_len), dtype=np.float32)
        x_inv_mask = np.zeros((self.total_length, max_num_neutrinos), dtype=bool)

        if direct_from_spanet:
            # In SPANet output, neutrino features are already aligned neutrino-wise
            for idx, (key, norm) in enumerate(self.event_info.generations['Neutrinos'].items()):
                if norm == "empty":
                    continue

                raw_feature = self.raw_data[f'INPUTS/Invisible/{key}']  # shape (n_events, max_num_neutrinos)

                if 'log' in norm:
                    x_inv[:, :, idx] = np.log1p(np.clip(raw_feature, a_min=1e-10, a_max=None))
                else:
                    x_inv[:, :, idx] = raw_feature

            x_inv_mask = np.ones((self.total_length, max_num_neutrinos), dtype=bool)
            num_valid_invisible = np.sum(x_inv_mask, axis=-1)
            num_raw_invisible = np.ones(self.total_length, dtype=np.int32) * max_num_neutrinos

        else:
            # REGRESSIONS/v/MASK path
            i_raw = 0
            for keys in self.raw_data.keys():
                if keys.startswith("REGRESSIONS") and "/v/MASK" in keys:
                    x_inv_mask[:, i_raw] = self.raw_data[keys]

                    for idx, (key, norm) in enumerate(self.event_info.generations['Neutrinos'].items()):
                        if norm == "empty":
                            continue

                        if 'log' in norm:
                            x_inv[:, i_raw, idx] = np.log1p(
                                np.clip(self.raw_data[keys.replace("/MASK", f"/{key}")], a_min=1e-10, a_max=None)
                            )
                        else:
                            x_inv[:, i_raw, idx] = self.raw_data[keys.replace("/MASK", f"/{key}")]

                    i_raw += 1
                    if i_raw >= max_num_neutrinos:
                        break

            num_raw_invisible = np.ones(self.total_length, dtype=np.int32) * i_raw
            num_valid_invisible = np.sum(x_inv_mask, axis=-1)

        return {
            "invisible-data": x_inv,
            "invisible-mask": x_inv_mask,
            "invisible-num-valid": num_valid_invisible,
            "invisible-num-raw": num_raw_invisible,
        }
