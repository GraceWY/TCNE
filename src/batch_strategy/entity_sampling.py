import os
import sys
import networkx as nx
import random
import numpy as np

from batch_strategy.alias_table_sampling import AliasTable as at

import pdb

INT = np.int32

class BatchStrategy(object):
    def __init__(self, G, params=None):
        """ G is a networkx with edge weight
        """
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.edges = G.edges()
        self.nodes = G.nodes()
        edge_probs = []
        node_probs = []
        

        # for sampling u and v_pos
        for e in self.edges:
            edge_probs.append(G[e[0]][e[1]]["weight"])
        self.edge_sampler = at(edge_probs)

        # for sampling v_neg
        #for n in self.nodes:
        #    node_probs.append(G.degree(n))
        #self.node_sampler = at(node_probs)

        degree = {}
        def dict_add(dic, k, v):
            if k in dic:
                dic[k] += v
            else:
                dic[k] = v
        for e in self.edges:
            dict_add(degree, e[0], G[e[0]][e[1]]['weight'])
            dict_add(degree, e[1], G[e[0]][e[1]]['weight'])

        for n in self.nodes:
            node_probs.append(degree[n])
        self.node_sampler = at(nodes_probs)

        self.conditional_node_sampler = {}
        self.conditional_node_lst = {}
        for n in self.nodes:
            tmp = []
            self.conditional_node_lst[n] = []
            for v in G[n]:
                tmp.append(degree[v])
                self.conditional_node_lst[n].append(v)
            self.conditional_node_sampler[n] = at(tmp)


    def get_batch(self):
        """
        """
        for _ in range(self.iterations):
            batch_u = []
            batch_v_pos = []
            batch_v_neg = []
            for _ in range(self.batch_size):
                idx = self.edge_sampler.sample()
                batch_u.append(self.edges[idx][0])
                batch_v_pos.append(self.edges[idx][1])
                idx = self.node_sampler.sample()
                batch_v_neg.append(self.nodes[idx])
            yield np.array(batch_u, dtype=INT), np.array(batch_v_pos, dtype=INT), np.array(batch_v_neg, dtype=INT)
