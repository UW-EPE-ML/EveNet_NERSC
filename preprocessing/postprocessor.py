import numpy as np
import torch
from decimal import Decimal, getcontext


def masked_stats(arr):
    mask = arr != 0
    values = np.where(mask, arr, 0)

    count = mask.sum(axis=0)
    sum_ = values.sum(axis=0)
    sumsq = (values ** 2).sum(axis=0)

    return {"sum": sum_, "sumsq": sumsq, "count": count}


def compute_effective_counts_from_freq(freqs: np.ndarray) -> np.ndarray:
    """
    Compute class-balanced weights based on effective number of samples.
    Ref: https://arxiv.org/pdf/1901.05555.pdf

    Args:
        freqs (np.ndarray): Array of sample counts per class. Index is class label.

    Returns:
        np.ndarray: Class weights normalized so that sum ≈ number of classes.
    """
    # TODO: check numerical stability

    freqs = freqs.astype(np.longdouble)
    N = freqs.sum()
    if N == 0:
        raise ValueError("Total number of samples is zero. Check input frequencies.")

    beta = 1 - (1 / N)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Avoid direct power to prevent underflow
        log_beta = np.log(beta)
        power_term = np.exp(freqs * log_beta)
        effective_num = (1.0 - power_term) / (1.0 - beta)

        weights = 1.0 / effective_num
        weights[~np.isfinite(weights)] = 0.0  # fix nan/inf
        weights = weights * len(freqs) / weights.sum()  # Normalize to total class count

    return weights


def compute_effective_counts_from_freq_decimal(freqs: list[float], precision: int = 50) -> np.ndarray:
    """
    Compute class-balanced weights based on effective number of samples using decimal for stability.

    Args:
        freqs (list[float]): List of sample counts per class.
        precision (int): Number of decimal places to use.

    Returns:
        list[float]: Class weights normalized to sum ≈ number of classes.
    """
    getcontext().prec = precision
    freqs = [Decimal(f) for f in freqs]
    N = sum(freqs)

    if N == 0:
        raise ValueError("Total number of samples is zero. Check input frequencies.")

    beta = Decimal(1) - Decimal(1) / N

    effective_num = []
    for f in freqs:
        if f == 0:
            effective_num.append(Decimal('Infinity'))  # or float('inf')
        else:
            numerator = Decimal(1) - beta ** f
            denominator = Decimal(1) - beta
            effective = numerator / denominator
            effective_num.append(effective)

    weights = [Decimal(1) / e if e != 0 else Decimal(0) for e in effective_num]
    total = sum(weights)
    normalized = [(w * len(freqs)) / total for w in weights]

    return np.array([float(w) for w in normalized], dtype=np.float32)


def compute_classification_balance(class_counts: np.ndarray) -> np.ndarray:
    """
    Wrapper to compute effective class weights from raw class frequency counts.
    """
    # return compute_effective_counts_from_freq(class_counts)
    return compute_effective_counts_from_freq_decimal(class_counts.tolist(), precision=50)


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
        "input_num": None,
    }

    for s in stats_list:
        for key in total.keys():
            if total[key] is None:
                total[key] = s[key]
            else:
                total[key] = merge_two(total[key], s[key])

    total['class_counts'] = np.sum([s["class_counts"] for s in stats_list], axis=0)

    # Final result
    result = {
        "x": compute_mean_std(total["x"]),
        "conditions": compute_mean_std(total["conditions"]),
        "regression": compute_mean_std(total["regression"]),
        "input_num": compute_mean_std(total["input_num"]),
        "class_counts": total["class_counts"],
        "class_balance": compute_classification_balance(total["class_counts"]),
    }
    return result


class PostProcessor:
    def __init__(self):
        self.stats = []

    def add(self, x, conditions, regression, num_vectors, class_counts):
        x_stats = masked_stats(x.reshape(-1, x.shape[-1]))
        cond_stats = masked_stats(conditions)
        regression_stats = masked_stats(regression)
        num_vectors_stats = masked_stats(num_vectors)
        self.stats.append({
            "x": x_stats,
            "conditions": cond_stats,
            "regression": regression_stats,
            "input_num": num_vectors_stats,
            "class_counts": class_counts,
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
            'class_counts': torch.tensor(merged_stats["class_counts"], dtype=torch.float32),
            'class_balance': torch.tensor(merged_stats["class_balance"], dtype=torch.float32),
        }

        if saved_results_path:
            torch.save(saved_results, f"{saved_results_path}/normalization.pt")
