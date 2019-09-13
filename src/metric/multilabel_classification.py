import os
import sys
import numpy as np
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
# from utils.draw_graph import DrawGraph as dg
from utils.lib_ml import MachineLearningLib as mll
from utils.env import *

import pdb


def params_handler(params, info, pre_res):
    if "res_home" not in params:
        params["res_home"] = info["res_home"]
    if "embeddings_file" in params:
        params["embeddings_path"] = os.path.join(info["home_path"], params["embeddings_file"])
    elif "infer" in pre_res and "entity_embedding_path" in pre_res["infer"]:
        params["embeddings_path"] = pre_res["infer"]["entity_embedding_path"]
    elif "optimize" in pre_res and "entity_embedding_path" in pre_res["optimize"]:
        params["embeddings_path"] = pre_res["optimize"]["entity_embedding_path"]

    if "ground_truth" not in params:
        params["ground_truth"] = os.path.join(info["network_folder"]["name"], info["network_folder"]["label"])

    if "file_type" not in params:
        params["file_type"] = "pickle"

    params["np_seed"] = info["np_seed"]

    return {}


def multilabel_classification(X, params):
    res = {}
    X_scaled = scale(X)
    y = dh.load_ground_truth(params["ground_truth"])
    pdb.set_trace()
    #print(len(y))
    #print("y_0=",y[0])
    ts = 0.0
    for i in range(9):
        ts += 0.1
        micro_f1 = 0.0
        macro_f1 = 0.0
        for _ in range(params["times"]):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = ts, random_state=params["np_seed"])
            # check whether is multi-label classification
            log = MultiOutputClassifier(SGDClassifier(loss='log'), n_jobs=params['n_jobs'])
            log.fit(X_train, y_train)

            for i in range(y_test.shape[1]):
                micro_f1 += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='micro')
                macro_f1 += f1_score(y_test[:, i], log.predict(X_test)[:, i], average='macro')

        micro_f1 /= (float(params["times"]*y.shape[1]))
        macro_f1 /= (float(params["times"]*y.shape[1]))
        print("test_size:",ts)
        res["%.2f" % ts] = {"micro_f1": micro_f1, "macro_f1": macro_f1}
        print({"micro_f1": micro_f1, "macro_f1": macro_f1})

    return res


@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)

    # load node number
    node_path=os.path.join(DATA_PATH, params["data"], "node.txt")
    node_file=open(node_path, 'r', encoding='utf-8', errors='ignore')
    nodes=node_file.readlines()
    node_num=len(nodes)
    node_file.close()

    # load embeddings 
    if params["file_type"] == "txt":
        embedding_path=os.path.join(DATA_PATH, "experiment", params["embeddings_path"])
        X = dh.load_embedding(embedding_path, params["file_type"], node_num)
    else:
        embedding_path = os.path.join(DATA_PATH, "experiment", params["embeddings_path"])
        X = dh.load_embedding(embedding_path, params["file_type"], node_num)

    # results include: accuracy, micro f1, macro f1
    metric_res = multilabel_classification(X, params)

    # insert into res
    for k, v in metric_res.items():
        res[k] = v

    return res
