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
