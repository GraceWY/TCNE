import os
import sys
import json
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.env import *

import pdb

def params_handler(params, info, pre_res, **kwargs):
    if ( "tag_walker" in pre_res ) and ( "walk_file" in pre_res["tag_walker"] ):
        params["walk_file"] = pre_res["tag_walker"]["walk_file"]

    params["en_embed_size"] = info["en_embed_size"]
    params["tag_embed_size"] = info["tag_embed_size"]

    return {}


@ct.module_decorator
def optimize(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)

    G_entity = dh.load_entity_as_graph(os.path.join(info["network_folder"]["name"], info["network_folder"]["edge"]), \
            os.path.join(info["network_folder"]["name"], info["network_folder"]["mix_edge"]), \
            os.path.join(info["network_folder"]["name"], info["network_folder"]["entity"]))  # G.node[id]["tags"] = binary lst tag 
    G_tag = dh.load_edge_as_graph(params["walk_file"], \
                os.path.join(info["network_folder"]["name"], info["network_folder"]["tag"])) # walk file
    params["en_num"] = len(G_entity.nodes()) 
    params["tag_num"] = len(G_tag.nodes())

    # get features
    gf_handler = __import__("get_features." + params["get_features"]["func"], fromlist = ["get_features"])
    features = gf_handler.get_features(params["get_feature"])

    # model init
    print ("[+] The embedding model is model.%s" % (params["embedding_model"]["func"]))
    info["logger"].info("[+] The embedding model is model.%s" % (params["embedding_model"]["func"]))
    model_handler = __import__("model." + params["embedding"]["func"], fromlist=["model"])
    model = model_handler.TagConditionedEmbedding(params, features)
    # load graph

    # get batch

    # init feature
