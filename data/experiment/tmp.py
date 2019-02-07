import numpy as np
import os
from sklearn.model_selection import train_test_split

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(FILE_PATH, "../..")
DATA_PATH = os.path.join(ROOT_PATH, "data")

path=os.path.join(DATA_PATH,"leetcode","label.txt")
file=open(path,'r')
lines=file.readlines()
print(lines)