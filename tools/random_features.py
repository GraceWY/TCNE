import numpy as np
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--num_nodes", type = int)
    parser.add_argument("--dim", type = int)
    args = parser.parse_args()
    fea = np.random.normal(size = [args.num_nodes, args.dim])
    with open("features.pkl", "wb") as f:
        pickle.dump(fea, f)
