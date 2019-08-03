import os
import sys
import networkx as nx
import random
import numpy as np

from batch_strategy.alias_table_sampling import AliasTable as at

import pdb

INT = np.int32
FLOAT = np.float64

class BatchStrategy(object):
    def __init__(self, G, params=None):
        """ G is a networkx with edge weight
        """
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.edges = G.edges()
        self.nodes = G.nodes()
        self.tag_scores = {}
        for i in range(len(G.node)):
            self.tag_scores[i] = G.node[i]["score"]
        edge_probs = []
        node_probs = []

        # pdb.set_trace()

        # for sampling u and v_pos
        for e in self.edges: 
            edge_probs.append(G[e[0]][e[1]]["weight"])
        self.edge_sampler = at(edge_probs)

        # for sampling v_neg
        for n in self.nodes:
            node_probs.append(G.degree(n))
        self.node_sampler = at(node_probs)


    def get_batch(self):
        """  
        """
        for _ in range(self.iterations):
            batch_u = []
            batch_v_pos = []
            batch_v_neg = []
            batch_u_score = []
            batch_v_pos_score = []
            for _ in range(self.batch_size):
                idx = self.edge_sampler.sample()
                batch_u.append(self.edges[idx][0])
                batch_v_pos.append(self.edges[idx][1])
                idx = self.node_sampler.sample()
                batch_v_neg.append(self.nodes[idx])
                batch_u_score.append(self.tag_scores[batch_u[-1]])
                batch_v_pos_score.append(self.tag_scores[batch_v_pos[-1]])
            yield np.array(batch_u, dtype=INT), np.array(batch_v_pos, dtype=INT), np.array(batch_v_neg, dtype=INT), np.array(batch_u_score, dtype=FLOAT), np.array(batch_v_pos_score, dtype=FLOAT)
