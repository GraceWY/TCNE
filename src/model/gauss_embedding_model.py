import os
import sys
import tensorflow as tf
import random

from utils import common_tools as ct

INT = tf.int32
FLOAT = tf.float32

# Gaussian Embedding
class NodeEmbedding(object):
    def __init__(self, params, mu=None, Sigma=None):
        self.dim = params["embed_size"]
        self.num_nodes = params["num_nodes"]
        self.covariance_type = params["covar_type"]
        self.optimizer = params["optimizer"]
        self.logger = params["logger"]

        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            tf.set_random_seed(random.randint(0, 1e9))
            self.u_id = tf.placeholder(INT, shape=[None])
            self.v_pos_id = tf.placeholder(INT, shape=[None])
            self.v_neg_id = tf.placeholder(INT, shape=[None])

            if mu is None:
                self.mu = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], -1.0/self.num_nodes, 1.0/self.num_nodes), dtype=FLOAT)
            else:
                self.mu = tf.Variable(mu, dtype=FLOAT)

            if Sigma is None:
                if self.covariance_type == "spherical":
                    self.Sigma = tf.Variable(tf.truncated_normal([self.num_nodes, 1], -1.0, 1.0), dtype=FLOAT)
                elif self.covariance_type == "diagonal":
                    self.Sigma = tf.Variable(tf.truncated_normal([self.num_nodes, self.dim], -1.0, 1.0), dtype=FLOAT)
            else:
                self.Sigma = tf.Variable(Sigma, dtype=FLOAT)






