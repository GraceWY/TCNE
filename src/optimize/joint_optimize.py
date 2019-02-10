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
    else:
        params["walk_file"] = os.path.join(info["home_path"], params["walk_file"])
    
    if ("optimize" in pre_res) and ("model_save_path" in pre_res["optimize"]):
        params["embedding_model"]["tag_embedding"]["tag_pre_train"] = pre_res["optimize"]["model_save_path"]
    elif "tag_pre_train" in params["embedding_model"]["tag_embedding"]:
        params["embedding_model"]["tag_embedding"]["tag_pre_train"] = os.path.join(
                info["home_path"],
                params["embedding_model"]["tag_embedding"]["tag_pre_train"])


    params["embedding_model"]["en_embed_size"] = info["en_embed_size"]
    params["embedding_model"]["tag_embed_size"] = info["tag_embed_size"]
    params["embedding_model"]["output_embed_size"] = info["output_embed_size"]
    params["embedding_model"]["res_home"] = info["res_home"]
    params["embedding_model"]["batch_size"] = params["batch_size"]
    params["embedding_model"]["show_num"] = params["show_num"]
    params["embedding_model"]["logger"] = info["logger"]

    return {}


@ct.module_decorator
def optimize(params, info, pre_res, **kwargs):
    #pdb.set_trace()
    res = params_handler(params, info, pre_res)

    G_entity = dh.load_entity_as_graph(os.path.join(info["network_folder"]["name"], info["network_folder"]["edge"]), \
            os.path.join(info["network_folder"]["name"], info["network_folder"]["mix_edge"]), \
            os.path.join(info["network_folder"]["name"], info["network_folder"]["entity"]))  # G.node[id]["tags"] = binary lst tag 
    G_tag = dh.load_edge_as_graph(params["walk_file"], \
                os.path.join(info["network_folder"]["name"], info["network_folder"]["tag"])) # walk file
    params["embedding_model"]["en_num"] = len(G_entity.nodes())
    params["embedding_model"]["tag_num"] = len(G_tag.nodes())
    info["en_num"] = params["embedding_model"]["en_num"]
    info["tag_num"] = params["embedding_model"]["tag_num"]

    # get features
    gf_handler = __import__("get_features." + params["get_features"]["func"], fromlist = ["sget_features"])

    if "dim" not in params["get_features"]:
        params["get_features"]["dim"] = params["tag_num"]

    features = gf_handler.get_features(params["get_features"], info)  # return numpy 

    # model init
    print ("[+] The embedding model is model.%s" % (params["embedding_model"]["func"]))
    info["logger"].info("[+] The embedding model is model.%s" % (params["embedding_model"]["func"]))

    params["embedding_model"]["aggregator"]["feature_num"] = params["get_features"]["dim"]

    model_handler = __import__("model." + params["embedding_model"]["func"], fromlist=["model"])
    model = model_handler.TagConditionedEmbedding(params["embedding_model"], features)
    model.build_graph()

    # batch generator
    print ("[+] The batch strategy is batch_strategy.%s" % (params["batch_strategy"]))
    info["logger"].info("[+] The batch strategy is batch_strategy.%s\n" % (params["batch_strategy"]))
    bs_handler = __import__("batch_strategy." + params["batch_strategy"], fromlist=["batch_strategy"])
    bs = bs_handler.BatchStrategy(G_tag, G_entity, params)

    # train model
    res["model_path"] = model.train(bs.get_batch)

    # infer model
    return res
