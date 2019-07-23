import os
import sys
import numpy as np
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh
from utils.draw_graph import DrawGraph as dg
from utils.env import *


def visualization(X, params):
    ground_truth_path=os.path.join(DATA_PATH,params["data"],params["ground_truth"]) 
    y = dh.load_ground_truth(ground_truth_path)
    y = y[:len(X)]

    row=len(X)
    column=len(X[0])

    if column>2:
        X=ct.reduce_embedding_dim(X,2)

    X=scale(X)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    cValue=ct.label2color(y)
    ax.scatter(X[:,0],X[:,1],c=cValue,cmap='viridis',marker='s')
    #plt.legend('x1')
    scatter_path = os.path.join(params["res_home"], params["embeddings_file"]+"scatter.pdf")
    plt.savefig(scatter_path)
    plt.show()

    return {"scatter_path" : scatter_path}


def params_handler(params, info):
    if "res_home" not in params:
        params["res_home"] = info["res_home"]
    return {}


@ct.module_decorator
def metric(params, info, pre_res, **kwargs):
    res = params_handler(params, info)

    # load embeddings 
    embedding_path=os.path.join(DATA_PATH,"experiment",params["embeddings_file"])   
    X = dh.load_embedding(embedding_path,params["file_type"])
    
    # results include: accuracy, micro f1, macro f1
    metric_res = visualization(X, params)

    # insert into res
    for k, v in metric_res.items():
        res[k] = v

    return res
