import os
import sys
import numpy as np
import pdb

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh

def params_handler(params, info, pre_res, **kwargs):
    res = {}
    res["entity_embedding_path"] = os.path.join(info["res_home"], "embeds.pkl")
    
    if "tag_mus_path" not in params:
        params["tag_mus_path"] = pre_res["optimize"]["mus"]

    return res

@ct.module_decorator
def infer(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    eid2tid_mat = dh.get_eid2tid(os.path.join(info["network_folder"]["name"], info["network_folder"]["mix_edge"]))
    embeds = dh.get_tagembed(params["tag_mus_path"], eid2tid_mat)
    dh.save_as_pickle(embeds, res["entity_embedding_path"])

    return res
