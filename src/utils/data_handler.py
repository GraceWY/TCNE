import os
import sys
import re
import networkx as nx
import json
import pickle
import numpy as np
from datetime import datetime

FLOAT = np.float32

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


    @staticmethod
    def load_dict(file_name):
        """ Load dict from file_name {key "\t" lst}
            Return mat and row2name which records the name of each row
        """
        mat = []
        row2name = dict()
        row = 0
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split("\t")
                row2name[row] = items[0]
                row += 1
                mat.append([float(i) for i in items[1].split()])
                
        return np.array(mat, dtype=FLOAT), row2name 
    
    @staticmethod
    def load_embedding(file_path,file_type,node_num=0):
        '''
        load embedding from file name 
        '''
        if file_type == "pickle":
            embedding = pickle.load(file_name)
        elif file_type == "txt":
            with open(file_path,'r') as f:
                lines = f.readlines()
                line_num=len(lines)
                node_number,dim=lines[0].split()
                node_number=int(node_number)
                dim=int(dim)
                embedding=np.zeros((node_num,dim))
                print(embedding.shape)
                for i in range(1,line_num):
                    x=lines[i].split()
                    if int(x[0])==11725: continue
                    if int(x[0])==0: print("found")
                    embedding[int(x[0]),:]=list(map(float,x[1:]))

        return embedding

    @staticmethod
    def load_ground_truth(file_name):
        '''load label for task node classification'''
        ground_truth_file=open(file_name,'r',encoding = 'gb2312')
        ground_truth=ground_truth_file.readlines()
        ground_truth_file.close()
        return ground_truth
        
if __name__ == "__main__":
    f = "tmp.txt"
    mp = {"1": [1, 1], "2": [2, 2]}
    DataHandler.save_dict(mp, f)
    
