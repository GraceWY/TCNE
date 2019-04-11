import os
import sys
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.draw_graph import DrawGraph as dg

import pdb




def params_handler(params, info):
    if "res_home" not in params:
        params["res_home"] = info["res_home"]

    if "timesOfSigma" not in params:
        params["timesOfSigma"] = 1 

    if "mus" in params:
        params["mus"] = os.path.join(params["path"], params["mus"])
        params["sigs"] = os.path.join(params["path"], params["sigs"])
    else:
        params["mus"] = pre_res["optimize"]["mus"]
        params["sigs"] = pre_res["optimize"]["sigs"]

    if "filter" not in params:
        params["filter"] = False

    if "draw_ellipse" not in params:
        params["draw_ellipse"] = true

    return {}


@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info)

    mus = dh.load_as_pickle(params["mus"])
    sigs = dh.load_as_pickle(params["sigs"])

    std_sigs = np.sqrt(sigs)

    row2name = dh.load_name(os.path.join(info["network_folder"]["name"], info["network_folder"]["tag"]))


    assert len(mus) > 0, "The mus file has no data"

    N = len(mus)
    M = len(mus[0])

    # sigs is spherical or diagonal
    if len(std_sigs[0]) == 1:
        ones = np.ones_like(mus)
        tmp = std_sigs.reshape(N, 1)
        std_sigs = ones*tmp

    # dimension reduction
    if M > 2:
        mus, std_sigs = ct.reduce_dist_dim(mus, std_sigs, 2)



    if params["draw_ellipse"]:
        res["res_path"] = os.path.join(params["res_home"], "dist_ellipse.pdf")
        dg.draw_ellipse(mus, std_sigs, row2name, res["res_path"], params["timesOfSigma"], params["filter"])
    else:
        res["res_path"] = os.path.join(params["res_home"], "dist_scatter.pdf")
        dg.draw_scatter(mus, std_sigs, row2name, res["res_path"], params["timesOfSigma"], params["filter"])


    res["scatter_path"] = os.path.join(params["res_home"], "scatter.pdf")
    dg.draw_scatter(mus, std_sigs, row2name, res["scatter_path"])

    return res
