import os
import sys
import numpy as np
import pickle
import io
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from networkx.readwrite import json_graph

sys.path.append('L:/GraphSAGE-master/graphsage')

model = 'concat'
time = 10
feat = False
prob = 0.1
name = 'cora'
tadw_path = 'tadw_'

def classification(X, y):
    X_scaled = scale(X)
    acc = 0.0
    micro_f1 = 0.0
    macro_f1 = 0.0
    for _ in range(time):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=prob, stratify=y)
        np.random.seed(1)
        log = SGDClassifier(loss="log", max_iter=500)
        log.fit(X_train, y_train)
        micro_f1 += f1_score(y_test, log.predict(X_test), average='micro')
        macro_f1 += f1_score(y_test, log.predict(X_test), average='macro')

    acc /= float(time)
    micro_f1 /= float(time)
    macro_f1 /= float(time)
    return {"acc": acc, "micro_f1": micro_f1, "macro_f1": macro_f1}


'''def params_handler(params, info):
    if "res_home" not in params:
        params["res_home"] = info["res_home"]
    return {}'''


# load embeddings

res = {}
labels = json.load(open(name + "-class_map.json"))
if model == 'graphsage':
    G = json_graph.node_link_graph(json.load(open(name + "-G.json")))
    #labels = json.load(open(name + "-class_map.json"))
    embeds = np.load("unsup-" + name + "\gcn_small_0.000010/val.npy")
    id_map = {}
    with open("unsup-" + name + "\gcn_small_0.000010/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[line.strip()] = i
    ids = [str(n) for n in G.nodes()]
    X = embeds[[id_map[id] for id in ids]]
    y = np.array([labels[str(i)] for i in ids])

elif model == 'tadw':
    import mat4py
    X = np.array(mat4py.loadmat(tadw_path + name + '/embedding.mat')['tmp'])
    y = np.array([x[1] for x in labels.items()])

elif model == 'concat':
    attr = np.load(name + '-feats.npy')
    G = json_graph.node_link_graph(json.load(open(name + "-G.json")))
    # labels = json.load(open(name + "-class_map.json"))
    embeds = np.load("unsup-" + name + "\gcn_small_0.000010/val.npy")
    id_map = {}
    with open("unsup-" + name + "\gcn_small_0.000010/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[line.strip()] = i
    ids = [str(n) for n in G.nodes()]
    #attr = attr[[id_map[id] for id in ids]]
    X = embeds[[id_map[id] for id in ids]]
    X = np.concatenate((X, attr), axis=1)
    y = np.array([labels[str(i)] for i in ids])


if feat:
    X = np.load(name + '-feats.npy')
# results include: accuracy, micro f1, macro f1
metric_res = classification(X, y)

# insert into res
for k, v in metric_res.items():
    res[k] = v

print(res)