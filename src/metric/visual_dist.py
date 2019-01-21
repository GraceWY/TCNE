import os
import sys
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.draw_graph import DrawGraph as dg

import pdb


def params_handler(params, info):
    if "res_home" in info:
        params["res_home"] = info["res_home"]

    if "timesOfSigma" not in params:
        params["timesOfSigma"] = 3

    return {}


@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info)

    mus, row2name = dh.load_dict(os.path.join(params["res_home"], "mus.dat"))
    sigs, row2name = dh.load_dict(os.path.join(params["res_home"], "sigs.dat"))
    std_sigs = np.sqrt(sigs)

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

    res["ellipse_path"] = os.path.join(params["res_home"], "dist_ellipse.pdf")

    dg.draw_ellipse(mus, std_sigs, row2name, res["ellipse_path"], params["timesOfSigma"])

    return res
