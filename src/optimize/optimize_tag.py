import os
import sys
import json
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.env import *

import pdb 


def params_handler(params, info, pre_res, **kwargs):
    # load training data
    if ( "tag_walker" in pre_res ) and ( "walk_file" in pre_res["tag_walker"] ):
        params["walk_file"] = pre_res["tag_walker"]["walk_file"]
    else:
        params["walk_file"] = os.path.join(info["res_home"], params["walk_file"])

    # set the embedding size
    params["embedding_model"]["tag_embed_size"] = info["tag_embed_size"]
    params["embedding_model"]["batch_size"] = params["batch_size"]
    params["embedding_model"]["logger"] = info["logger"]

    return {}


@ct.module_decorator
def optimize(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)

    G = dh.load_edge_as_graph(params["walk_file"], os.path.join(info["network_folder"]["name"], info["network_folder"]["tag"]))
    params["embedding_model"]["num_nodes"] = len(G.nodes())
    params["embedding_model"]["res_home"] = info["res_home"]

    pdb.set_trace()

    # model init
    print ("[+] The embedding model is model.%s" % (params["embedding_model"]["func"]))
    info["logger"].info("[+] The embedding model is model.%s\n" % (params["embedding_model"]["func"]))
    model_handler = __import__("model." + params["embedding_model"]["func"], fromlist = ["model"])
    # model = model_handler.NodeEmbedding(params["embedding_model"], features)
    model = model_handler.NodeEmbedding(params["embedding_model"])

    # get_batch generator
    print ("[+] The batch strategy is batch_strategy.%s" % (params["batch_strategy"]))
    info["logger"].info("[+] The batch strategy is batch_strategy.%s\n" % (params["batch_strategy"]))
    bs_handler = __import__("batch_strategy." + params["batch_strategy"], fromlist=["batch_strategy"])
    bs = bs_handler.BatchStrategy(G, params)

    # train model
    res["model_save_path"], mus, logsigs = model.train(bs.get_batch)
    sigs = np.exp(logsigs)

    res["mus"] = os.path.join(info["res_home"], "mus.pkl")
    res["sigs"] = os.path.join(info["res_home"], "sigs.pkl")

    # save result
    dh.save_as_pickle(mus, res["mus"])
    dh.save_as_pickle(sigs, res["sigs"])

    return res
