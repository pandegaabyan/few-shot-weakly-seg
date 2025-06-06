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
