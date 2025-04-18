import numpy as np
from collections import OrderedDict
from evenet.control.event_info import EventInfo


class EveNetFromSPANetDataConverter:
    def __init__(
            self,
            raw_data: dict,
            event_info: EventInfo,
            process: str,
    ):

        self.raw_data = raw_data
        self.event_info = event_info
        self.process = process
        self.total_length = None

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

            self.total_length = features_ak.shape[0]
        return output_dict

    def load_assignments(self, assignment_names, assignment_map):
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

            target_prefix = f"{label}/{product}"
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


def preprocess(in_dir, store_dir, process_info, unique_id, cfg_dir=None, save: bool = True):
    converted_data = []

    # if hasattr(config, 'event_info'):
    if not global_config.loaded:
        global_config.load_yaml(cfg_dir)

    converted_statistics = PostProcessor(global_config)
    assignment_keys, assignment_key_map = generate_assignment_names(global_config.event_info)
    regression_keys, regression_key_map = generate_regression_names(global_config.event_info)
    unique_process_ids = sorted(set(v['process_id'] for v in global_config.process_info.values()))

    shape_metadata = None

    for process in global_config.process_info:
        # for process in ["TT1L", "TT2L", "WZ_3L"]:
        # print("Processing ", process)
        matched_data = monitor_gen_matching(
            in_dir=in_dir,
            process=process,
            feynman_diagram_process=global_config.process_info[process],
            monitor_plots=False,
        )

        if matched_data is None:
            print(f"[WARNING] No matched data for process {process} in dir {in_dir}")
            continue

        converter = EveNetDataConverter(
            raw_data=deepcopy(matched_data),
            event_info=global_config.event_info,
            process=process,
        )

        # Filter the data
        converter.filter_process(process=process, process_info=process_info[process])

        # Load Point Cloud and Mask
        sources = converter.load_sources()
        num_sequential_vectors = np.sum(sources['sources-0-mask'], axis=1)
        num_vectors = num_sequential_vectors + np.sum(sources['sources-1-mask'][:, np.newaxis], axis=1)

        assignments = converter.load_assignments(assignment_keys, assignment_key_map)
        regressions = converter.load_regressions(regression_keys, regression_key_map)
        classifications = converter.load_classification()

        invisible = converter.load_invisible(max_num_neutrinos=global_config.get("max_neutrinos", 2))

        process_data = {
            'num_vectors': num_vectors,
            'num_sequential_vectors': num_sequential_vectors,
            'subprocess_id': converter.raw_data['METADATA/PROCESS'],

            'x': sources['sources-0-data'],
            'x_mask': sources['sources-0-mask'],
            'conditions': sources['sources-1-data'],
            'conditions_mask': sources['sources-1-mask'][:, np.newaxis],

            'classification': classifications['classification-EVENT/signal'],

            'x_invisible': invisible['invisible-data'],
            'x_invisible_mask': invisible['invisible-mask'],
            'num_invisible_raw': invisible['invisible-num-raw'],
            'num_invisible_valid': invisible['invisible-num-valid'],

            **assignments,
            **regressions,
        }

        flattened_data, meta_data = flatten_dict(process_data)
        # Count the number of unique processes
        # class_counts = np.bincount(process_data['classification'], minlength=len(unique_process_ids))
        # Or simply use the process ID and set the length of the array to the number of unique processes
        class_counts = np.zeros(len(unique_process_ids), dtype=np.int32)
        class_counts[process_info[process]['process_id']] = len(process_data['classification'])

        total_subprocess = global_config.event_info.event_mapping
        subprocess_counts = np.zeros(len(total_subprocess), dtype=np.int32)
        if process in total_subprocess:
            subprocess_counts[list(total_subprocess.keys()).index(process)] = len(process_data['classification'])

        assignment_mask_per_process = {}
        assignment_idx = {key: i for i, key in enumerate(assignment_keys) if f'TARGETS/{process}/' in key}
        for key, i in assignment_idx.items():
            particle = key.split('/')[2]
            assignment_mask_per_process[particle] = assignments['assignments-mask'][:, i]

        if shape_metadata is None:
            shape_metadata = meta_data
        else:
            assert (shape_metadata == meta_data), "Shape metadata mismatch"

        print(f"[INFO] Current table size: {flattened_data.nbytes / 1024 / 1024:.2f} MB")

        converted_data.append(flattened_data)

        # Calculating normalization statistics
        converted_statistics.add(
            x=process_data['x'],
            conditions=process_data['conditions'],
            regression=process_data['regression-data'],
            num_vectors=process_data['num_sequential_vectors'],
            class_counts=class_counts,
            subprocess_counts=subprocess_counts,
            invisible=process_data['x_invisible'],
        )
        # Add assignment mask
        converted_statistics.add_assignment_mask(process, assignment_mask_per_process)

    if len(converted_data) == 0:
        print(f"[WARNING] No data found for any of the processes in {in_dir}")
        return None

    final_table = pa.concat_tables(converted_data)

    shuffle_indices = np.random.default_rng(31).permutation(final_table.num_rows)
    final_table = final_table.take(pa.array(shuffle_indices))

    if save:
        ### Save to parquet
        pq.write_table(final_table, f"{store_dir}/data_{unique_id}.parquet")

        with open(f"{store_dir}/shape_metadata.json", "w") as f:
            json.dump(shape_metadata, f)

        print(f"[INFO] Final table size: {final_table.nbytes / 1024 / 1024:.2f} MB")
        print(f"[Saving] Saving {final_table.num_rows} rows to {store_dir}/data_{unique_id}.parquet")

        return converted_statistics

    else:
        return converted_statistics, final_table, shape_metadata


if __name__ == '__main__':
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('preprocess_config', help='Path to config file', default='preprocess_pretrain.yaml')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--store_dir', type=str, default='Storage')
    parser.add_argument(
        '--pretrain_dirs', type=str, nargs='+',
        help='Pretrain directories, accept a list of directories, will force using 2-level directory structure'
    )
    args = parser.parse_args()

    main(args)

