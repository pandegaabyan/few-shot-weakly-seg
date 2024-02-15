def get_iso_timestamp_now() -> str:
    import datetime

    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def convert_iso_timestamp_to_epoch(iso_timestamp: str) -> float:
    import datetime

    datetime_obj = datetime.datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S")
    return datetime.datetime.timestamp(datetime_obj)


def convert_epoch_to_iso_timestamp(epoch: float, use_utc: bool = False) -> str:
    import datetime

    if use_utc:
        datetime_obj = datetime.datetime.utcfromtimestamp(epoch)
    else:
        datetime_obj = datetime.datetime.fromtimestamp(epoch)
    return datetime_obj.isoformat()


def convert_local_iso_to_utc_iso(iso_timestamp: str) -> str:
    import datetime

    datetime_obj = datetime.datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S")
    timestamp = datetime.datetime.timestamp(datetime_obj)
    return datetime.datetime.utcfromtimestamp(timestamp).isoformat()


def merge_dicts(dicts: list[dict]) -> dict:
    return {k: v for d in dicts for k, v in d.items()}


def get_deep(obj, key: str):
    try:
        splitted_key = key.split(".")
        if len(splitted_key) > 1:
            first_key = (
                splitted_key[0] if isinstance(obj, dict) else int(splitted_key[0])
            )
            return get_deep(obj[first_key], ".".join(splitted_key[1:]))
        else:
            return obj[key]
    except (KeyError, IndexError):
        return None


def diff_dict(old_dict: dict, new_dict: dict, should_process: bool = True) -> dict:
    from deepdiff import DeepDiff
    from deepdiff.model import PrettyOrderedSet

    if not should_process:
        return dict(DeepDiff(old_dict, new_dict))

    def standardize_value_type(dc: dict[str, dict]):
        for group in dc.values():
            for key, value in group.items():
                if isinstance(value, tuple):
                    group[key] = list(value)

    def simplify_key(key: str) -> str:
        return (
            key.removeprefix("root['")
            .removesuffix("']")
            .replace("']['", ".")
            .replace("]['", ".")
            .replace("'][", ".")
            .replace("[]", ".")
        )

    standardize_value_type(old_dict["config"])
    standardize_value_type(new_dict["config"])

    diff = dict(DeepDiff(old_dict, new_dict))

    new_diff = {}
    for name, item in diff.items():
        new_name = name.split("_")[-1]
        if isinstance(item, dict):
            new_item = {}
            for key, value in item.items():
                new_item[simplify_key(key)] = value
            new_diff[new_name] = new_item
        elif isinstance(item, PrettyOrderedSet):
            new_item = {}
            for key in item:
                simplified_key = simplify_key(key)
                if new_name == "added":
                    ref_value = get_deep(new_dict, simplified_key)
                elif new_name == "removed":
                    ref_value = get_deep(old_dict, simplified_key)
                else:
                    ref_value = None
                new_item[simplified_key] = ref_value
            new_diff[new_name] = new_item
        else:
            new_diff[new_name] = item
    return new_diff


def parse_string(s: str) -> bool | int | float | str:
    if s == "True":
        return True
    elif s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def make_batch_sample_indices(
    population_size: int, sample_size: int, batch_size: int
) -> list[list[int]]:
    import random

    samples = sorted(random.sample(range(population_size), sample_size))
    population_batch_size = population_size // batch_size + 1
    batch_samples = [[] for _ in range(population_batch_size)]
    for s in samples:
        batch_samples[s // batch_size].append(s - (s // batch_size) * batch_size)
    return batch_samples


def generate_char(idx: int) -> str:
    if idx < 0 or idx > 26 * 26:
        raise ValueError("idx must be between 0 and 26*26")
    if idx < 26:
        return chr(65 + idx)
    else:
        return chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
