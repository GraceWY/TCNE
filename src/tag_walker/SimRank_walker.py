import itertools
import numpy
import networkx as nx
import os

from utils.data_handler import DataHandler as dh
from tag_walker import utils

EPS = 1e-4
DIV_EPS = 1e-8
CHUNK_SIZE = 2000


def params_handler(params, info, pre_res, **kwargs):
    return {}

def tag_walker(params, info, pre_res, **kwargs):
    """ Return the filepath with the written walks \
            [from_tag, to_tag, weight] = (str, str, int)

        params: the params of this module
        info: the whole params of the model
        pre_res: the results from the previous modules, for this case is HG, G
    """
    
    res = params_handler(params,info,pre_res)

    prefix = "simrank_iter(%d)_eps(%d)_r(%d)" % (params["max_iter"], params["eps"],params["r"])
    res["walk_file_details"] = os.path.join(info["network_folder"]["name"], "%s_details.dat"%(prefix))
    res["walk_file"] = os.path.join(info["network_folder"]["name"], "%s.dat"%(prefix))


    #res["walk_file"] = os.path.join(info["network_folder"]["name"], "SimRank_walker.dat")
    #res["walk_file_details"] = os.path.join(info["network_folder"]["name"], "SimRank_walker_details.dat")
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

    tag_lst = [i for i, n in HG.nodes(data=True) if n["type"] == "tag"]
    out_mapp = dict()

    nodes=HG.nodes()
    nodes_dic=HG.nodes(data=True)
    nodes_i = {nodes[i]: i for i in range(0, len(nodes))}

    sim_prev = numpy.zeros(len(nodes))
    sim = numpy.identity(len(nodes))

    max_iter=params["max_iter"]
    eps=params["eps"]
    r=params["r"]

    for i in range(max_iter):
        if numpy.allclose(sim, sim_prev, atol=eps): 
            break
    sim_prev = numpy.copy(sim)
    for u, v in itertools.product(nodes, nodes):
        if u is v: continue
        u_ps, v_ps = HG.neighbors(u), HG.neighbors(v)
        s_uv = sum(sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ps, v_ps))
        sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ps) * len(v_ps) + DIV_EPS)

    len_nodes=len(nodes)
    for i in range(len_nodes):
        for j in range(i+1,len_nodes):
            if nodes_dic[i][1]["type"]=="tag" and nodes_dic[j][1]["type"]=="tag":
                tag_pair=utils.get_simrank_output(HG,nodes[i],nodes[j])
                out_mapp[tag_pair]=sim[i][j]


    with open(res["walk_file"], "w") as f:
        for k, v in out_mapp.items():
            f.write("%s\t%.8f\n" % (k, v))
    
    info["logger"].info("the tag similarity file path is: %s" % (res))

    return res
