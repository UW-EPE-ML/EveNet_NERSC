from evenet.dataset.types import Batch, Source, AssignmentTargets


def process_event_batch(batch: list[dict]) -> dict:
    out=process_event(batch)
    return {"batch": out}


def process_event(row) -> Batch:
    # Build sources
    sources = []
    i = 0

    while f"sources-{i}-data" in row:
        sources.append(Source(data=row[f"sources-{i}-data"], mask=row[f"sources-{i}-mask"]))
        i += 1

    # num_vectors
    num_vectors = row["num_vectors"]

    # assignment_targets
    assignment_targets = []
    for key in row:
        if key.startswith("assignments-") and key.endswith("-indices"):
            parts = key.split("-")
            process = parts[1]
            particle = parts[2]
            base = f"assignments-{process}-{particle}"
            indices = row[f"{base}-indices"]
            mask = row[f"{base}-mask"]
            assignment_targets.append(AssignmentTargets(indices=indices, mask=mask))

    # regression_targets
    regression_targets = {}
    for key in row:
        if key.startswith("regression-") and key.endswith("-data"):
            name = key[len("regression-"):-len("-data")]
            base = f"regression-{name}"
            regression_targets[name] = (
                row[f"{base}-data"],
                row[f"{base}-mask"]
            )

    # classification_targets
    classification_targets = {}
    for key in row:
        if key.startswith("classification-"):
            name = key[len("classification-"):]
            classification_targets[name] = row[key]

    # num_sequential_vectors
    num_sequential_vectors = {}
    if "num_sequential_vectors" in row:
        num_sequential_vectors["Source"] = row["num_sequential_vectors"]

    return {"batch": Batch(
        sources=tuple(sources),
        num_vectors=num_vectors,
        assignment_targets=tuple(assignment_targets),
        regression_targets=regression_targets,
        classification_targets=classification_targets,
        num_sequential_vectors=num_sequential_vectors,
    )}
