import os
import sys
import json
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.env import *

import pdb 

def params_handler(params, info,  **kwargs):

    return {}


def get_feature(params, info, **kwargs):
    res = params_handler(params)
    return np.ramdom.normal(size = [params["dim"], info["en_num"]])


