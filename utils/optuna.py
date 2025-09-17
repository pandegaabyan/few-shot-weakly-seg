import os

import optuna
from dotenv import load_dotenv

from config.constants import FILENAMES
from utils.logging import check_mkdir
from utils.utils import parse_string


def get_optuna_storage(
    dummy: bool = False, engine_kwargs: dict | None = None
) -> optuna.storages.RDBStorage:
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
        url=db_url, heartbeat_interval=30 * 60, engine_kwargs=engine_kwargs
    )


def load_study(study_id: str, dummy: bool = False) -> optuna.Study | None:
    storage = get_optuna_storage(dummy, engine_kwargs={"pool_size": 1})
    try:
        study_names = optuna.get_all_study_names(storage)
        study_name = next(filter(lambda x: x.endswith(study_id), study_names))
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return None
    finally:
        storage.engine.dispose()


def get_study_best_name(
    study_id: str, dummy: bool = False
) -> tuple[str | None, str | None]:
    study = load_study(study_id, dummy)
    if study is None or len(study.trials) == 0:
        return None, None
    return study.best_trial.user_attrs.get("run_name"), study.user_attrs.get("exp_name")


def parse_hyperparams(hparams: str) -> dict[str, bool | int | float | str]:
    return {
        key.strip(): parse_string(value.strip())
        for hp in hparams.strip("[]").split(",")
        for key, value in [hp.split(":")]
    }
