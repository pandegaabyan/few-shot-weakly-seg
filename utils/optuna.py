import os

import optuna
from dotenv import load_dotenv

from config.constants import FILENAMES
from utils.logging import check_mkdir


def get_optuna_storage(
    dummy: bool = False, engine_kwargs: dict | None = None
) -> optuna.storages.BaseStorage:
    if dummy:
        log_dir = FILENAMES["log_folder"]
        check_mkdir(log_dir)
        db_url = f"sqlite:///{log_dir}/optuna_dummy.sqlite3"
    else:
        load_dotenv()
        db_url = os.getenv("OPTUNA_DB_URL")
        if not db_url:
            raise ValueError("OPTUNA_DB_URL is not set")
    return optuna.storages.RDBStorage(
        url=db_url, heartbeat_interval=5 * 60, engine_kwargs=engine_kwargs
    )
