import os
import sys
import numpy as np
import pdb

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh

def params_handler(params, info, pre_res, **kwargs):
    res = {}
    res["entity_embedding_path"] = os.path.join(info["res_home"], "embeds.pkl")
    return res

@ct.module_decorator
def infer(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    embeds = dh.get_tagonehot(os.path.join(info["network_folder"]["name"], info["network_folder"]["mix_edge"]))
    dh.save_as_pickle(embeds, res["entity_embedding_path"])

    return res
