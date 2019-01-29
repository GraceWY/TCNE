import os
import sys
import networkx as nx
import random
import numpy as np

from batch_strategy.alias_table_sampling import AliasTable as at
from batch_strategy.weighted_edges_sampling import BacthStrategy as bs_tag
from batch_strategy.entity__sampling import BacthStrategy as bs_en

import pdb

INT = np.int32

class BatchStrategy(object):
    def __init__(self, G_tag, G_entity, params=None):
        """ G is a networkx with edge weight
        """
        self.iterations = params["iterations"]

        self.bs_tag_handler = bs_tag(G_tag, params)
        self.bs_entity_handler = bs_en(G_entity,params)


    def get_batch(self):
        """
        """
        def deal_res(tag_res, en_res):
            dic = {"en_" + k, v for k, v in en_res.items()}
            dic["tag_u"] = tag_res[0]
            dic["tag_v"] = tag_res[1]
            dic["tag_n"] = tag_res[2]
            return dic
        tag_res = bs_tag_handler.send(None)
        en_res = bs_entity_handler.send(None)
        yield deal_res(tag_res, en_res)
        for _ in range(self.iterations - 1):
            tag_res = next(bs_tag_handler)
            en_res = next(bs_en_handler)
            yield deal_res(tag_res, en_res)
