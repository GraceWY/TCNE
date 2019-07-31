import os
import sys
import networkx as nx
import random
import numpy as np

from utils.data_handler import DataHandler as dh
from utils import common_tools as ct
from tag_walker import utils

import pdb

def params_handler(params, info, **kwargs):
    return {}

def get_similarity_reverse(diff):
    return 1/(diff+1)

def get_similarity_exp(diff):
    return np.exp(-diff)

@ct.module_decorator
def tag_walker(params, info, pre_res, **kwargs):
    """ Return the filepath with the written walks \
            [from_tag, to_tag, similarity] = (str, str, int)

        params: the params of this module
        info: the whole params of the model
        pre_res: the results from the previous modules, for this case is HG, G
    """
    res = params_handler(params, info)
    res["walk_file"] = os.path.join(info["network_folder"]["name"], "shortest_path.dat")
    res["walk_file_details"] = os.path.join(info["network_folder"]["name"], "shortest_path_details.dat")
    # assert not os.path.exists(res["walk_file"]), "the walk file has existed!"

    # if the file has existed, return directly
    if os.path.exists(res["walk_file"]):
        print ("[+] The walk_file has existed in %s \n \
                this module %s has finished!" % (res["walk_file"], params["func"]))
        info["logger"].info("[+] The walk_file has existed in %s \n \
                this module %s has finished!" % (res["walk_file"], params["func"]))
        return res

    HG = pre_res["construct_graph"]["HG"]
    EG = pre_res["construct_graph"]["G"]

    tag_lst = [i for i, n in HG.nodes(data=True) \
            if n["type"] == "tag"]

    out_mapp = dict()
    with open(res["walk_file_details"], "w") as f:
        for i in range(len(tag_lst)):
            for j in range(i+1, len(tag_lst)):
                nei_i = HG.neighbors(tag_lst[i])
                nei_j = HG.neighbors(tag_lst[j])
                len_lst = []
                for ni in nei_i:
                    for nj in nei_j:
                        try:
                            path = nx.shortest_path(EG, source=ni, target=nj)
                            if len(path)+1 > params["max_path_len"]:
                                continue

                            npath = [tag_lst[i]]
                            npath = npath + path
                            npath.append(tag_lst[j])
                            len_lst.append(len(npath)-1)
                            f.write(utils.get_output_details(HG, EG, npath))
                        except nx.NetworkXNoPath:
                            continue

                if len(len_lst) > 0:
                    tag_pair = utils.get_output(HG, [tag_lst[i], tag_lst[j]])
                    out_mapp[tag_pair] = get_similarity_exp(sum(len_lst) / float(len(len_lst)))
                    # out_mapp[tag_pair] = get_similarity_exp(min(len_lst))

    with open(res["walk_file"], "w") as f:
        for k, v in out_mapp.items():
            f.write("%s\t%.8f\n" % (k, v))
    
    info["logger"].info("the tag similarity file path is: %s" % (res))

    return res


## I'm not sure whether tag can be plugged into the path between tag and tag
## and in this version, other tags can be plugged in the inner path.
def tag_walker_old(params, info, pre_res, **kwargs):
    """ Return the filepath with the written walks \
            [from_tag, to_tag, weight] = (str, str, int)

        params: the params of this module
        info: the whole params of the model
        pre_res: the results from the previous modules, for this case is HG, G
    """
    res = params_handler(params, info)
    res["walk_file"] = os.path.join(info["network_folder"]["name"], "shortest_path.dat")
    res["walk_file_details"] = os.path.join(info["network_folder"]["name"], "shortest_path_details.dat")
    assert not os.path.exists(res["walk_file"]), "the walk file has existed!"

    HG = pre_res["construct_graph"]["HG"]
    EG = pre_res["construct_graph"]["G"]

    # merge the two graph into G
    G = nx.Graph()
    G.add_nodes_from(HG.nodes(data=True) + EG.nodes(data=True))
    G.add_edges_from(HG.edges(data=True) + EG.edges(data=True))

    tag_lst = [i for i, n in HG.nodes(data=True) \
            if n["type"] == "tag"]

    out_mapp = dict()
    with open(res["walk_file_details"], "w") as f:
        for i in range(len(tag_lst)):
            for j in range(i+1, len(tag_lst)):
                path = nx.shortest_path(G, source=tag_lst[i], target=tag_lst[j])
                f.write(utils.get_output_details(HG, EG, path))
                tag_pair = utils.get_output(HG, path)
                out_mapp[tag_pair] = len(path)-1

    with open(res["walk_file"], "w") as f:
        for k, v in out_mapp.items():
            f.write(k + "\t" + str(v))

    return res
