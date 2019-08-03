import os, sys
import math
import pandas as pd
import pdb
import numpy as np
from wordfreq import zipf_frequency

TOPIC_RATIO=0.5
THETA=2

# load data
data_path = "../../data/leetcode"

fn_mix_edge = os.path.join(data_path, "mix_edge.dat")
fn_tag = os.path.join(data_path, "tag.dat")
fn_entity = os.path.join(data_path, "entity.dat")

# generate data filename
fn_tag_features = os.path.join(data_path, "tag_features.csv")
fn_training_data = os.path.join(data_path, "training.csv")


def get_num(fn):
    cnt = 0
    with open(fn, "rb") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                break
            cnt += 1
    return cnt


def tag_id_map(fn):
    tag2num = {}
    num2tag = {}
    with open(fn, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) == 0:
                continue
            num = int(items[-1])
            tag = " ".join(items[:-1])
            assert tag not in tag2num
            tag2num[tag] = num
            num2tag[num] = tag

    return tag2num, num2tag

def map_tag_entity(fn, tag_num, en_num):
    tag2en = {}
    for i in range(tag_num):
        tag2en[i] = set()
    en2tag = {}
    for i in range(en_num):
        en2tag[i] = set()

    edge_weight = {}

    with open(fn, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 2:
                continue
            eid = int(items[0])
            tid = int(items[1])
            w = 1
            if len(items) == 3:
                w = int(items[2])

            en2tag[eid].add(tid)
            tag2en[tid].add(eid)
            
            if eid not in edge_weight:
                edge_weight[eid] = {}
            edge_weight[eid][tid] = w

    return tag2en, en2tag, edge_weight

def get_tag_TF(tag_num, tag2id, id2tag):
    ret = [0.0]*tag_num
    for key in tag2id:
        ret[tag2id[key]] = zipf_frequency(key, "en")
        # print (key, ret[tag2id[key]])

    print ("get tag tf over!")
    return ret


def get_tag_IDF(tag_num, tag2en, edge_weight):
    ret = [0.0]*tag_num
    for tag in tag2en:
        for en in tag2en[tag]:
            assert en in edge_weight and tag in edge_weight[en]
            ret[tag] += edge_weight[en][tag]
 
    print ("get tag idf over!")
    return ret


def get_pr(tag, topic, dict_tf_of_tag, tag2en, edge_weight):
    down = dict_tf_of_tag[topic]
    assert down > 0, "down is not bigger than 0."
    top = 0

    inter_set = tag2en[tag] & tag2en[topic]

    for en in inter_set:
        top += min(edge_weight[en][tag], edge_weight[en][topic])

    return top*1.0/down


def get_tag_entropy(tag_num, tag2en, edge_weight):
    # get top tag:
    dict_tf_of_tag = {}
    for tag in tag2en:
        for en in tag2en[tag]:
            if tag not in dict_tf_of_tag:
                dict_tf_of_tag[tag] = 0
            dict_tf_of_tag[tag] += edge_weight[en][tag]
    
    lst_tf_of_tag = [(id, dict_tf_of_tag[id]) for id in dict_tf_of_tag]
    lst_tf_of_tag.sort(key=lambda x:x[1], reverse=True)

    topic_num = round(TOPIC_RATIO*tag_num)

    # implement p(en, tag)
    tag_entropy = [0.0]*tag_num
    for tag in range(tag_num):
        for j in range(topic_num):
            topic = lst_tf_of_tag[j][0]
            pr = get_pr(tag, topic, dict_tf_of_tag, tag2en, edge_weight)
            if pr > 0:
                tag_entropy[tag] += pr*math.log(pr)
        tag_entropy[tag] = -tag_entropy[tag]

    print ("get tag entropy over!")
    return tag_entropy


def save_tag_features():
    tag2id, id2tag = tag_id_map(fn_tag)
    tag_num = get_num(fn_tag)
    tag_tf = get_tag_TF(tag_num, tag2id, id2tag)

    en_num = get_num(fn_entity)
    tag2en, en2tag, edge_weight = map_tag_entity(fn_mix_edge, tag_num, en_num)
    tag_idf = get_tag_IDF(tag_num, tag2en, edge_weight)

    tag_entropy = get_tag_entropy(tag_num, tag2en, edge_weight)

    tag_feas = np.array(list(zip(tag_tf, tag_idf, tag_entropy)))
    # pdb.set_trace()

    df = pd.DataFrame({"tf": tag_feas[:, 0],
            "idf": tag_feas[:, 1],
            "entropy": tag_feas[:, 2]}, 
            columns= ["tf", "idf", "entropy"])

    df.to_csv(fn_tag_features, index=False)
    
    print ("save tag features over!")


def load_tag_features(fn):
    return pd.read_csv(fn, dtype=np.float64).to_numpy()



def get_D(tag_i, tag_j, tag2en, edge_weight): # return float
    # pdb.set_trace()
    D = 0
    # get intersection
    inter_st = tag2en[tag_i] & tag2en[tag_j]
    for en in inter_st:
        if edge_weight[en][tag_i] > edge_weight[en][tag_j]:
            D += (edge_weight[en][tag_i] - edge_weight[en][tag_j])

    res_st = tag2en[tag_i] - inter_st
    for en in res_st:
        D += edge_weight[en][tag_i]

    return D


def save_training_data():
    tag_num = get_num(fn_tag)
    en_num = get_num(fn_entity)
    tag2en, en2tag, edge_weight = map_tag_entity(fn_mix_edge, tag_num, en_num)

    tag_feas = load_tag_features(fn_tag_features)
    
    #pdb.set_trace()

    ret = []
    for i in range(tag_num):
        for j in range(i+1, tag_num):
            dij = get_D(i, j, tag2en, edge_weight) # return float
            dji = get_D(j, i, tag2en, edge_weight)
            if dij == 0 or dji == 0:
                continue
            if dij*1.0/dji > THETA:
                tmp_pos = np.array(tag_feas[i]) - np.array(tag_feas[j])
                tmp_pos = np.append(tmp_pos, [1])
                tmp_neg = -tmp_pos
                ret.append(tmp_pos)
                ret.append(tmp_neg)

            if dij*1.0/dji < 1.0/THETA:
                tmp_neg = np.array(tag_feas[i]) - np.array(tag_feas[j])
                tmp_neg = np.append(tmp_neg, [-1])
                tmp_pos = -tmp_neg
                ret.append(tmp_neg)
                ret.append(tmp_pos)

    ret = np.array(ret)
    df = pd.DataFrame({"delta_tf": ret[:, 0],
        "delta_idf": ret[:, 1],
        "delta_entropy": ret[:, 2],
        "y": ret[:, 3]}, columns= ["delta_tf", "delta_idf", "delta_entropy", "y"])

    df.to_csv(fn_training_data, index=False)
    print ("get training data over!")


if __name__ == "__main__":
     save_tag_features()
     save_training_data()






