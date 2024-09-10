def mean(ls: list[float]) -> float:
    if len(ls) == 0:
        return 0
    return sum(ls) / len(ls)


def merge_dicts(dicts: list[dict]) -> dict:
    return {k: v for d in dicts for k, v in d.items()}


def parse_string(s: str):
    import ast

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def make_batch_sample_indices(
    population_size: int, sample_size: int, batch_size: int
) -> list[list[int]] | None:
    import random

    if sample_size == 0:
        return None
    random.seed(0)
    samples = sorted(random.sample(range(population_size), sample_size))
    population_batch_size = population_size // batch_size + 1
    batch_samples = [[] for _ in range(population_batch_size)]
    for s in samples:
        batch_samples[s // batch_size].append(s - (s // batch_size) * batch_size)
    random.seed(None)
    return batch_samples
