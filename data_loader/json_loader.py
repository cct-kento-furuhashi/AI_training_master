import json
import os
from typing import Dict, Union

from settings.params import PARAM_PATH, TRAIN_PARAMS


def read_param() -> Dict[str, Union[int, float, str]]:
    params: Dict[str, Union[int, float, str]] = TRAIN_PARAMS
    if os.path.exists(PARAM_PATH):
        with open(PARAM_PATH) as rf:
            params = json.load(rf)
    return params


def save_param(params: Dict[str, Union[int, float, str]]) -> None:
    with open(PARAM_PATH, "w") as wf:
        json.dump(params, wf)