import os
import sys
import re
import networkx as nx
import json
import numpy as np
from datetime import datetime

class DataHandler(object):
    @staticmethod
    def load_edge(file_path):
        """ 
            Read edge file {int, int [, weight]}
            Return:
                edge list with weight [int, int, float]
        """
        lst = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) == 2:
                    items.append(1) # the weight of edge
                items[0] = int(items[0]) # from node
                items[1] = int(items[1]) # to node
                items[2] = float(items[2]) # edge weight
                lst.append(items)
        return lst

    @staticmethod
    def load_name(file_path):
        """
            Load name file, which map str to id {str, id}

            Return:
                reversed dict {id, str} (node_id, name)
        """
        mp = dict()
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                mp[" ".join(items[:-1])] = int(items[-1])
        reverse_mp = dict(zip(mp.values(), mp.keys()))
        return reverse_mp

    @staticmethod
    def load_json(file_path):
        """
            Load json file
        """
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s', "", s)
        return json.loads(s)

    @staticmethod
    def load_as_graph(file_path):
        """ Load walk file {str, str, float}
            Return networkx G
        """
        lst = []
        mp = dict()
        G = nx.Graph()
        cur = 0
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split("\t")
                if items[0] not in mp:
                    mp[items[0]] = cur
                    G.add_node(cur, {"name": items[0]})
                    cur += 1
                if items[1] not in mp:
                    mp[items[1]] = cur
                    G.add_node(cur, {"name": items[1]})
                    cur += 1
                w = 1.0 if len(items) < 3 else float(items[2])
                G.add_edge(mp[items[0]], mp[items[1]], weight=w)
        return G


    @staticmethod
    def save_dict(mp, file_name):
        """ Save dict to file_name
        """
        with open(file_name, "w") as f:
            for k, v in mp.items():
                line = k + "\t" +  " ".join([str(i) for i in mp[k]]) + "\n"
                f.write(line)


if __name__ == "__main__":
    f = "tmp.txt"
    mp = {"1": [1, 1], "2": [2, 2]}
    DataHandler.save_dict(mp, f)
    
