import os
import sys
from random import randrange, random
import numpy as np

FLOAT = np.float64
INT = np.int32

class AliasTable():
    def __init__(self, _probs):
        probs = np.array(_probs, dtype=FLOAT)
        self.num = len(_probs)
        probs = self.num * probs / np.sum(probs)
        print (probs)
        self.probs_table = np.ones(self.num, dtype=FLOAT)
        self.alias_table = np.zeros(self.num, dtype=INT)
        L, H = [], []

        for i in range(self.num):
            if probs[i] == 1:
                self.probs_table[i] = 1.0
                self.alias_table[i] = None
            elif probs[i] < 1:
                L.append(i)
            else:
                H.append(i)

        while len(L) > 0 and len(H) > 0:
            l = L.pop()
            h = H.pop()
            self.probs_table[l] = probs[l]
            self.alias_table[l] = h
            probs[h] = probs[h] - (1 - probs[l])
            if probs[h] == 1:
                self.probs_table[h] = 1.0
                self.alias_table[h] = None
            elif probs[h] < 1:
                L.append(h)
            else:
                H.append(h)
        del L, H

    def sample(self):
        idx = randrange(self.num)
        if random() < self.probs_table[idx]:
            return idx
        else:
            return self.alias_table[idx]

if __name__ == "__main__":
    test = [1, 1, 3, 2, 4]
    at = AliasTable(test)
    mapp = {}
    for i in range(1000000):
        t = at.sample()
        if t in mapp:
            mapp[t] += 1
        else:
            mapp[t] = 1
    print (mapp)

