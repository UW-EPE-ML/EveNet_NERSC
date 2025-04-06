import numpy as np
import torch


def masked_stats(arr):
    mask = arr != 0
    values = np.where(mask, arr, 0)

    count = mask.sum(axis=0)
    sum_ = values.sum(axis=0)
    sumsq = (values ** 2).sum(axis=0)

    return {"sum": sum_, "sumsq": sumsq, "count": count}


def merge_stats(stats_list):
    def merge_two(a, b):
        return {
            "sum": a["sum"] + b["sum"],
            "sumsq": a["sumsq"] + b["sumsq"],
            "count": a["count"] + b["count"]
        }

    def compute_mean_std(agg):
        count = agg["count"]
        sum_ = agg["sum"]
        sumsq = agg["sumsq"]

        # Avoid divide-by-zero
        safe_count = np.where(count == 0, 1, count)

        mean = sum_ / safe_count
        variance = sumsq / safe_count - mean ** 2
        variance = np.clip(variance, a_min=0.0, a_max=None)
        std = np.sqrt(variance)

        # Set mean = 0, std = 1 for features with no data
        mean = np.where(count == 0, 0.0, mean)
        std = np.where(count == 0, 1.0, std)
        std = np.where(std == 0, 1.0, std)

        return {'mean': mean, 'std': std}

    # Accumulate across all files
    total = {
        "x": None,
        "conditions": None,
        "regression": None,
        'input_num': None
    }

    for s in stats_list:
        for key in total.keys():
            if total[key] is None:
                total[key] = s[key]
            else:
                total[key] = merge_two(total[key], s[key])

    # Final result
    result = {
        "x": compute_mean_std(total["x"]),
        "conditions": compute_mean_std(total["conditions"]),
        "regression": compute_mean_std(total["regression"]),
        "input_num": compute_mean_std(total["input_num"]),
    }
    return result


class PostProcessor:
    def __init__(self):
        self.stats = []

    def add(self, x, conditions, regression, num_vectors):
        x_stats = masked_stats(x.reshape(-1, x.shape[-1]))
        cond_stats = masked_stats(conditions)
        regression_stats = masked_stats(regression)
        num_vectors_stats = masked_stats(num_vectors)
        self.stats.append({
            "x": x_stats,
            "conditions": cond_stats,
            "regression": regression_stats,
            "input_num": num_vectors_stats,
        })

    @classmethod
    def merge(
            cls,
            instances,
            regression_names,
            saved_results_path=None,
    ):
        # Filter out None instances, when a run dir does not contain any data 
        # for the desired physics processes
        valid_instances = [inst for inst in instances if inst is not None]
        combined = [item for a in valid_instances for item in a.stats]
        merged_stats = merge_stats(combined)

        saved_results = {
            'input_mean': {
                'Source': torch.tensor(merged_stats["x"]["mean"], dtype=torch.float32),
                'Conditions': torch.tensor(merged_stats["conditions"]["mean"], dtype=torch.float32),
            },
            'input_std': {
                'Source': torch.tensor(merged_stats["x"]["std"], dtype=torch.float32),
                'Conditions': torch.tensor(merged_stats["conditions"]["std"], dtype=torch.float32),
            },
            'input_num_mean': {
                'Source': torch.tensor(merged_stats["input_num"]["mean"], dtype=torch.float32)
            },
            'input_num_std': {
                'Source': torch.tensor(merged_stats["input_num"]["std"], dtype=torch.float32)
            },
            'regression_mean': {
                k: torch.tensor(merged_stats["regression"]["mean"][i], dtype=torch.float32)
                for i, k in enumerate(regression_names)
            },
            'regression_std': {
                k: torch.tensor(merged_stats["regression"]["std"][i], dtype=torch.float32)
                for i, k in enumerate(regression_names)
            },
        }

        if saved_results_path:
            torch.save(saved_results, f"{saved_results_path}/normalization.pt")
