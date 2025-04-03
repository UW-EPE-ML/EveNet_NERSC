import numpy as np


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

        # Avoid divide-by-zero by substituting 1 where count is 0 (doesn't matter because we mask later)
        safe_count = np.where(count == 0, 1, count)

        mean = sum_ / safe_count
        std = np.sqrt(sumsq / safe_count - mean ** 2)

        # Zero out mean/std where count == 0
        mean = np.where(count == 0, 0.0, mean)
        std = np.where(count == 0, 0.0, std)

        return {'mean': mean, 'std': std}

    # Accumulate across all files
    total = {
        "x": None,
        "conditions": None,
        "regression-data": None,
    }

    for s in stats_list:
        for key in ["x", "conditions", "regression-data"]:
            if total[key] is None:
                total[key] = s[key]
            else:
                total[key] = merge_two(total[key], s[key])

    # Final result
    result = {
        "x": compute_mean_std(total["x"]),
        "conditions": compute_mean_std(total["conditions"]),
        "regression-data": compute_mean_std(total["regression-data"]),
    }
    return result

