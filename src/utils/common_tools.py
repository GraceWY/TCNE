import os
import sys
import time
import logging
from datetime import datetime
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import pdb


def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")


def symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        os.remove(dst)
        os.symlink(src, dst)


def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):
    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, "w")
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(("%(asctime)s %(filename)s" \
            "[line:%(lineno)d] %(levelname)s %(message)s"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def module_decorator(func):
    """ The decorator of moduler
    """
    def wrapper(*args, **kwargs):
        print ("[+] Start %s ..." % (kwargs["mdl_name"], ))
        kwargs["info"]["logger"].info("[+] Start %s ...\n" % (kwargs["mdl_name"], ))
        start_time = datetime.now()

        res = func(*args, **kwargs)

        end_time = datetime.now()

        duration = (end_time-start_time).seconds

        print ("[+] Finished !\n[+] During Time: %.2f" % (duration))
        kwargs["info"]["logger"].info("[+] Finished !\n[+] During Time: %.2f\n" % (duration))

        res["Duration"] = duration
        print ("[+] Module Results: %s\n" % (str(res)))
        kwargs["info"]["logger"].info("[+] Module Results: %s\n\n" % (str(res)))

        return res
    return wrapper


def reduce_dist_dim(mus, std_sigs, dim):
    """ Todo with PCA
    """
    pca = PCA(n_components=dim)
    pca.fit(mus)
    de_mus = pca.transform(mus)
    de_std_sigs = pca.transform(std_sigs)
    return de_mus, de_std_sigs


def glorot_init(shape, _dtype, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=_dtype)
    return tf.Variable(initial, name=name)

if __name__ == "__main__":
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    nX = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    x, y = reduce_dist_dim(X, nX, 2)
    print (x)
    print (y)
