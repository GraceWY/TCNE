# -*- coding: utf-8 -*-
import os
import sys
import re
import networkx as nx
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

import pdb

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
        with open(file_path, "r", encoding="gb2312") as f:
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
    def load_edge_as_graph(file_path, name_path):
        """ Load walk file {int, int, float}
            name path: {string, id}
            Return networkx G
        """
        lst = []
        G = nx.Graph()
        id2name = DataHandler.load_name(name_path)
        for k, v in id2name.items():
            G.add_node(k, {"name": v})
        
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                fid = int(items[0])
                # G.add_node(fid, {"name": id2name[fid]})
                tid = int(items[1])
                # G.add_node(tid, {"name": id2name[tid]})
                w = 1.0 if len(items) < 3 else float(items[2])
                G.add_edge(fid, tid, weight=w)
        return G


    @staticmethod
    def load_entity_as_graph(entity_edge_path, entity2tag_path, entity_name_path, tag_name_path):
        """ entity_path: {int , int} (fid, tid)
            tag_entity_path: {int, int} (entity_id, tag_id)
            entity_name_path: {str, int} (entity name, id)
            name_path: {str, int} (tag name, id)

            Return entity networkx G with tag_id_binary_vector as G.nodes[id]["tags"]
        """
        G = DataHandler.load_edge_as_graph(entity_edge_path, entity_name_path)
        tag_id2name = DataHandler.load_name(tag_name_path)
        tag_num = len(tag_id2name)
        
        # load tag 01 vector for each entity
        mp = dict()
        with open(entity2tag_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                eid, tid = int(items[0]), int(items[1])

                if eid not in mp:
                    mp[eid] = []
                mp[eid].append(tid)

        lst = [[] for k in range(len(G.nodes()))] 
        for k in mp.keys():
            lst[k] = mp[k]
        
        #mlb = MultiLabelBinarizer()
        #mat = mlb.fit_transform(lst)

        for n in G.nodes():
            G.node[n]["tags"] = np.zeros(tag_num, dtype = np.int32)
            for i in lst[n]:
                G.node[n]["tags"][i] = 1
        
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
            with open(file_path, "rb") as fn:
                embedding = pickle.load(fn)
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


    @staticmethod
    def save_as_pickle(data, file_name):
        with open(file_name, "wb") as fn:
            pickle.dump(data, fn)

    @staticmethod
    def load_as_pickle(file_name):
        with open(file_name, "rb") as fn:
            return pickle.load(fn)

    @staticmethod
    def get_eid2tid(file_name):
        '''
            Load {entity \t tag} file
            Return the matrix where i-th row is a list including the tag ids 
        '''
        e2t = {}
        node_num = 0
        with open(file_name, "r") as fn:
            for line in fn:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                it1, it2 = int(items[0]), int(items[1])

                if it1 not in e2t:
                    e2t[it1] = []
                    if node_num < it1:
                        node_num = it1
                e2t[it1].append(it2)

        node_num += 1

        #pdb.set_trace()
        #print ("node_num: %d", node_num)
        #print ("dict_size: %d", len(e2t))

        lst = [[] for k in range(node_num)]

        for k in e2t.keys():
            lst[k] = e2t[k]

        return lst

    @staticmethod
    def get_tagonehot(file_name):
        '''
            Load mix edge (entity \t tag) to a dictionary
            Key=entity, Value={tag id}

            Return tag one hot embedding as each node embedding
        '''
        lst = DataHandler.get_eid2tid(file_name)

        mlb = MultiLabelBinarizer()
        mat = mlb.fit_transform(lst)

        return mat

    @staticmethod
    def get_tagembed(file_name, e2t_mat):
        """
            Load tag embedding file and e2t_mat
            Return node embedding as the average of relative tag embeddings
        """

        # Load tag embedding
        with open(file_name, "rb") as fn:
            tag_embed = np.array(pickle.load(fn))

        assert len(tag_embed) > 0, "The tag embedding file is null."

        
        # for these node without tags
        u_embed = np.mean(tag_embed, axis=0)

        node_embed = []

        for l in e2t_mat:
            if len(l) == 0:
                node_embed.append(u_embed)
            else:
                tmp = tag_embed[l, :]
                tmp_m = np.mean(tag_embed[l, :], axis=0)
                if len(np.shape(tmp_m)) != 1:
                    pdb.set_trace()
                node_embed.append(np.mean(tag_embed[l, :], axis=0))

        return node_embed

if __name__ == "__main__":
    folder = "/Users/wangyun/repos/TCNE/data/lc" 
    entity_edge_path = os.path.join(folder, "edge.dat")
    entity2tag_path = os.path.join(folder, "mix_edge.dat")
    entity_name_path = os.path.join(folder, "entity.dat")
    DataHandler.load_entity_as_graph(entity_edge_path, entity2tag_path, entity_name_path)
