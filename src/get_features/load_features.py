import os
import sys
import json
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.env import *

import pdb 

def params_handler(params, info,  **kwargs):
    params["file_path"] = os.path.join(info["data_path"], info["network_folder"]["name"])
    params["file_path"] = os.path.join(params["file_path"],
            info["network_folder"]["entity_features"])
    return {}


def get_features(params, info, **kwargs):
    res = params_handler(params, info)
    return  dh.load_as_pickle(params["file_path"])
