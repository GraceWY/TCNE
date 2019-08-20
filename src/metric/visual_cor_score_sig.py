import os
import sys
import numpy as np
import pandas as pd

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.draw_graph import DrawGraph as dg

import pdb




def params_handler(params, info):
    if "res_home" not in params:
        params["res_home"] = info["res_home"]

    params["tag_score"] = os.path.join(info["network_folder"]["name"], info["network_folder"]["tag_score"])

    if "sigs" in params:
        params["sigs"] = os.path.join(params["path"], params["sigs"])
    else:
        params["sigs"] = pre_res["optimize"]["sigs"]

    return {}


@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info)

    sigs = dh.load_as_pickle(params["sigs"])
    std_sigs = np.sqrt(sigs)

    tag_scores = dh.load_list(params["tag_score"])

    assert len(std_sigs) > 0, "The std_sigs file has no data"

    #pdb.set_trace()
    # init dataframe
    data_dict = {}
    name_lst = []
    data_dict["tag_score"] = tag_scores
    name_lst.append("tag_score")
    if len(std_sigs.shape) == 1:
        data_dict["std_sig"] = std_sigs
        name_lst.append("std_sig")
    else:
        #data_dict["mean_std_sig"] = np.mean(std_sigs, axis=1)
        data_dict["product_std_sig"] = 0.1*np.product(std_sigs, axis=1)
        name_lst.append("product_std_sig")
        """
        for i in range(len(std_sigs[0])):
            data_dict["std_sig"+str(i)] = std_sigs[:, i]
            name_lst.append("std_sig"+str(i))
        """
    df = pd.DataFrame(data_dict, columns=name_lst)

    res["res_path"] = os.path.join(params["res_home"], "cor_score_stdsig.pdf")

    if "plot_kind" not in params:
        dg.draw_lines(df, res["res_path"])
    else:
        dg.draw_lines(df, res["res_path"], params["plot_kind"])



    return res
