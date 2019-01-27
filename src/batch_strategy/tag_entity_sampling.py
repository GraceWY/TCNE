import os
import sys
import networkx as nx
import random
import numpy as np

from batch_strategy.alias_table_sampling import AliasTable as at

import pdb

INT = np.int32

class BatchStragegy(object):
    def __init__(self, EG, TG, params=None):
        """ EG is an entity node networkx with weight
            TG is an tag node networkx with weight
        """
        self.iterations = params["iterations"]
        self.en_batch_size = params["en_batch_size"]
        self.tag_batch_size = params["tag_batch_size"]

        self.
