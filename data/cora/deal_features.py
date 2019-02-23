from sklearn.preprocessing import MultiLabelBinarizer
import os
import numpy as np
import pickle

node = [[] for _ in range(2708)]
with open("tag-edge.txt", "r") as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        items = line.split()
        node[int(items[0])].append(int(items[1]))
    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(node)

with open("features_tag.pkl", "wb") as f:
    pickle.dump(mat, f)
