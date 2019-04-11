import os
import sys
import numpy as np
import shutil
import random
import pickle

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh


def params_handler(params, info, pre_res, **kwargs):
    res = {}
    if ct.check_attr(params, "save_path", info["res_home"]):
        params["save_path"] = os.path.join(info["home_path"], params["save_path"])
    params["folder_path"] = os.path.join(info["data_path"], info["network_folder"]["name"])
    res["train_path"] = os.path.join(params["save_path"], "train")
    res["test_path"] = os.path.join(params["save_path"], "test")
    ct.mkdir(res["train_path"])
    ct.mkdir(res["test_path"])
    return res

def line_init(line, number = None):
    line = line.strip()
    if len(line) == 0:
        return None
    items = line.split()
    if number is not None and len(items) != number:
        return None
    return items

@ct.module_decorator
def graph_split(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)
    for k, v in info["network_folder"].items():
        if k == "name":
            continue
        shutil.copy(os.path.join(params["folder_path"], v), res["train_path"])
        shutil.copy(os.path.join(params["folder_path"], v), res["test_path"])
    #folder_info = {k: os.path.join(info["home_path"], v) for k, v in info["network_folder"].items()}
    folder_info = ct.obj_dic(info["network_folder"])
    node_list = []
    with open(os.path.join(params["folder_path"], folder_info.entity), "r", encoding="gb2312") as f:
        for line in f:
            items = line_init(line, 2)
            if items is None:
                continue
            node_list.append((int(items[1]), items[0]))
    
    n = len(node_list)
    n_test = int(float(n) * params["test_ratio"])
    n_train = n - n_test
    
    random.shuffle(node_list)
    node_dic = {it[0]: [idx, it[1]] for idx, it in enumerate(node_list)}
    # write entity file
    f_test = open(os.path.join(res["test_path"], folder_info.entity), "w", encoding="gb2312")
    f_train = open(os.path.join(res["train_path"], folder_info.entity), "w", encoding="gb2312")
    for k, v in node_dic.items():
        f_test.write("%s %d\n" % (v[1], v[0]))
        if v[0] < n_train:
            f_train.write("%s %d\n" % (v[1], v[0]))
    f_test.close()
    f_train.close()
    
    # write edge file
    f_test = open(os.path.join(res["test_path"], folder_info.edge), "w")
    f_train = open(os.path.join(res["train_path"], folder_info.edge), "w")
    with open(os.path.join(params["folder_path"], folder_info.edge), "r") as f:
        for line in f:
            items = line_init(line, 2)
            if items is None:
                continue
            it = [node_dic[int(i)][0] for i in items]
            f_test.write("%d %d\n" % (it[0], it[1]))
            if it[0] < n_train and it[1] < n_train:
                f_train.write("%d %d\n" % (it[0], it[1]))
    for i in range(n_train):
        f_test.write("%d %d\n" % (i, i))
        f_train.write("%d %d\n" % (i, i)) 
    for i in range(n_train, n):
        f_test.write("%d %d\n" % (i, i))
        
    
    f_test.close()
    f_train.close()
    
    # write mix_edge file
    f_test = open(os.path.join(res["test_path"], folder_info.mix_edge), "w")
    f_train = open(os.path.join(res["train_path"], folder_info.mix_edge), "w")
    with open(os.path.join(params["folder_path"], folder_info.mix_edge), "r") as f:
        for line in f:
            items = line_init(line, 2)
            if items is None:
                continue
            items[0] = node_dic[int(items[0])][0]
            f_test.write("%d %s\n" % (items[0], items[1]))
            if items[0] < n_train:
                f_train.write("%d %s\n" % (items[0], items[1]))
    f_test.close()
    f_train.close()
    
    # write label
    f_test = open(os.path.join(res["test_path"], folder_info.label), "w", encoding="gb2312")
    f_train = open(os.path.join(res["train_path"], folder_info.label), "w", encoding="gb2312")
    label_list = []
    with open(os.path.join(params["folder_path"], folder_info.label), "r", encoding="gb2312") as f:
        for line in f:
            items = line_init(line, 1)
            if items is None:
                continue
            label_list.append(items[0])
    for i in range(n_train):
        f_test.write("%s\n" % label_list[node_list[i][0]])
        f_train.write("%s\n" % label_list[node_list[i][0]])
    
    for i in range(n_train, n):
        f_test.write("%s\n" % label_list[node_list[i][0]])
    f_test.close()
    f_train.close()
    
    #write entity_features
    with open(os.path.join(params["folder_path"], folder_info.entity_features), "rb") as f:
        features = pickle.load(f)
    
    idx_list, _ = zip(*node_list)
    with open(os.path.join(res["test_path"], folder_info.entity_features), "wb") as f:
        pickle.dump(features[idx_list, :], f)
    with open(os.path.join(res["train_path"], folder_info.entity_features), "wb") as f:
        pickle.dump(features[idx_list[:n_train], :], f)
        
    return res
