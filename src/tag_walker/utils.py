import os
import sys

def get_output_details(HG, G, path):
    for i in path[1:-1]:
        assert i in G, str(i) + " | " + " ".join([str(i) for i in path])
    return HG.node[path[0]]["name"] + "\t" + " ".join([G.node[i]["name"] for i in path[1:-1]]) + "\t" + HG.node[path[-1]]["name"] + "\n"


def get_output(HG, path):
    tag1, tag2 = HG.node[path[0]]["oid"], HG.node[path[-1]]["oid"]
    if tag1 >= tag2:
        tag1, tag2 = tag2, tag1
    return str(tag1) + "\t" + str(tag2)

def get_simrank_output(HG,node1,node2):
    tag1, tag2 = HG.node[node1]["oid"], HG.node[node2]["oid"]
    if tag1 >= tag2:
        tag1, tag2 = tag2, tag1
    return str(tag1) + "\t" + str(tag2)
