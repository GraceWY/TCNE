import os
import sys
import networkx as nx
import random
import numpy as np

from batch_strategy.alias_table_sampling import AliasTable as at

import pdb

INT = np.int32
FLOAT = np.float32

class BatchStrategy(object):
    def __init__(self, G, params=None):
        """ G is a networkx with edge weight
        """
        self.G = G
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.nce_k = params["embedding_model"]["generative_net"]["nce_k"]
        self.tag_num = params["embedding_model"]["tag_num"]
        self.tag_embed_size = params["embedding_model"]["tag_embed_size"]
        self.agg_neighbor_num = params["embedding_model"]["aggregator"]["agg_neighbor_num"]
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

        # for sampling v_neg
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
        self.node_sampler = at(node_probs)

        # for sampling neighbors
        self.conditional_node_sampler = {}
        self.conditional_node_lst = {}
        for n in self.nodes:
            tmp = []
            self.conditional_node_lst[n] = []
            for v in G[n]:
                tmp.append(degree[v])
                self.conditional_node_lst[n].append(v)
            self.conditional_node_sampler[n] = at(tmp)

    def sample_unit(self, u):
        """
        """
        mask = [self.G.node[u]["tags"][i] for i in range(self.tag_num)]
        noise = np.random.normal(size = [self.tag_num, self.tag_embed_size])
        neighbors = []
        for i in range(self.agg_neighbor_num):
            idx = self.conditional_node_sampler[u].sample()
            neighbors.append(self.conditional_node_lst[u][idx])
        return mask, noise, neighbors

    def get_batch(self):
        """
        """
        for _ in range(self.iterations):
            batch_u = []
            batch_u_mask = []
            batch_u_noise = []
            batch_u_neighbors = []
            batch_v = []
            batch_v_mask = []
            batch_v_noise = []
            batch_v_neighbors = []
            batch_n = []
            batch_n_mask = []
            batch_n_noise = []
            batch_n_neighbors = []
            for _ in range(self.batch_size):
                idx = self.edge_sampler.sample()
                u, v = self.edges[idx][0], self.edges[idx][1]
                batch_u.append(u)
                batch_v.append(v)
                mask, noise, neighbors = self.sample_unit(u)
                batch_u_mask.append(mask)
                batch_u_noise.append(noise)
                batch_u_neighbors.append(neighbors)
                
                mask, noise, neighbors = self.sample_unit(v)
                batch_v_mask.append(mask)
                batch_v_noise.append(noise)
                batch_v_neighbors.append(neighbors)

                for _ in range(self.nce_k):
                    idx = self.node_sampler.sample()
                    n = self.nodes[idx]
                    batch_n.append(n)
                    mask, noise, neighbors = self.sample_unit(n)
                    batch_n_mask.append(mask)
                    batch_n_noise.append(noise)
                    batch_n_neighbors.append(neighbors)

                
            yield {"u" : np.array(batch_u, dtype=INT),
                    "u_mask" : np.array(batch_u_mask, dtype=FLOAT),
                    "u_noise" : np.array(batch_u_noise, dtype=FLOAT),
                    "u_neighbors" : np.array(batch_u_neighbors, dtype = INT),
                    "v" : np.array(batch_v, dtype=INT),
                    "v_mask" : np.array(batch_v_mask, dtype=FLOAT),
                    "v_noise" : np.array(batch_v_noise, dtype=FLOAT),
                    "v_neighbors" : np.array(batch_v_neighbors, dtype = INT),
                    "n" : np.array(batch_n, dtype=INT),
                    "n_mask" : np.array(batch_n_mask, dtype=FLOAT),
                    "n_noise" : np.array(batch_n_noise, dtype=FLOAT),
                    "n_neighbors" : np.array(batch_n_neighbors, dtype = INT)}
