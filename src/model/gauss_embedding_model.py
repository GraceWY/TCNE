import os
import sys
import tensorflow as tf
import random
import math
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh

import pdb

INT = tf.int32
FLOAT = tf.float32

# Gaussian Embedding
class NodeEmbedding(object):
    def __init__(self, params):
        self.dim = params["embed_size"]
        self.num_nodes = params["num_nodes"]
        self.spherical = params["spherical"]
        self.show_num = params["show_num"] # printing the loss per #show_num iterations
        self.batch_size = params["batch_size"]

        # The constant in margin-based loss
        self.Closs = 1.0 if "Closs" not in params else params["Closs"]
        # whether to fix the variance or not, that makes the var untrainable
        self.fixvar = False if "fixvar" not in params else params["fixvar"] 
        # whether the w and c use the uniform parameters
        self.wout = False if "wout" not in params else params["wout"] 

        # whether to clip the range of mu and sigma for distribution
        self.normclip = False if "normclip" not in params else params["normclip"]

        """ lower_sig: element-wise lower bound for sigma
            upper_sig: element-wise upper bound for sigma
            norm_cap: upper bound of norm of mu
        """
        self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = 0.02, 5.0, 3.0, 1, 0.05
        if "lower_sig" in params:
            self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = params["lower_sig"], params["upper_sig"], params["norm_cap"], params["mu_scale"], params["var_scale"] 

        self.optimizer = params["optimizer"]
        self.lr = params["learning_rate"]
        self.logger = params["logger"]


    def build_graph(self):
        """ Build graph for Gauss Embedding
        """
        mu_scale = self.mu_scale * math.sqrt(3.0/(1.0 * self.dim))
        logvar_scale = math.log(self.var_scale)
        var_trainable = 1-self.fixvar
        lower_logsig = math.log(self.lower_sig)
        upper_logsig = math.log(self.upper_sig)
        self.tensor_graph = tf.Graph()
        with self.tensor_graph.as_default():
            tf.set_random_seed(random.randint(0, 1e9))
            with tf.name_scope("GaussEmbedding"):
                with tf.name_scope("input"):
                    self.u_id = tf.placeholder(name="u_id", dtype=INT, shape=[None])
                    self.v_pos_id = tf.placeholder(name="vpos_id", dtype=INT, shape=[None])
                    self.v_neg_id = tf.placeholder(name="vneg_id", dtype=INT, shape=[None])

                with tf.name_scope("Variable"):
                    self.mu = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], \
                            -mu_scale, mu_scale), name="mu", dtype=FLOAT)

                    if self.wout:
                        self.mu_out = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], \
                                -mu_scale, mu_scale), name="mu_out", dtype=FLOAT)

                    if self.spherical:
                        self.logsig = tf.Variable(tf.random_uniform([self.num_nodes, 1], \
                                logvar_scale, logvar_scale), name="logsigma", dtype=FLOAT, trainable=var_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.num_nodes, 1], \
                                    logvar_scale, logvar_scale), name="logsigma_out", dtype=FLOAT, trainable=var_trainable)

                    else:
                        self.logsig = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], \
                                logvar_scale, logvar_scale), name="logsigma", dtype=FLOAT, trainable=var_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.num_nodes, self.dim], \
                                    logvar_scale, logvar_scale), name="logsigma_out", dtype=FLOAT, trainable=var_trainable)

                    if not self.wout:
                        self.mu_out, self.logsig_out = self.mu, self.logsig

                    self.mu_embed = tf.nn.embedding_lookup(self.mu, self.u_id) 
                    self.mu_embed_pos = tf.nn.embedding_lookup(self.mu_out, self.v_pos_id) 
                    self.mu_embed_neg = tf.nn.embedding_lookup(self.mu_out, self.v_neg_id) 
                    self.sig_embed = tf.exp(tf.nn.embedding_lookup(self.logsig, self.u_id))
                    self.sig_embed_pos = tf.exp(tf.nn.embedding_lookup(self.logsig_out, self.v_pos_id))
                    self.sig_embed_neg = tf.exp(tf.nn.embedding_lookup(self.logsig_out, self.v_neg_id))

                def energy(mu_i, sig_i, mu_j, sig_j):
                    """E(P[i], P[j]) = Negative KL divergence
                    """
                    N = self.batch_size 
                    if self.spherical:
                        trace_fac = self.dim * sig_j / sig_i
                        det_fac = self.dim * tf.log(sig_j / sig_i)
                    else:
                        trace_fac = tf.reshape(tf.reduce_sum(sig_j / sig_i, axis=1), [N, 1]) 
                        det_fac = tf.reshape(tf.reduce_sum(tf.log(sig_j) - tf.log(sig_i), axis=1), [N, 1])

                    return -0.5 * (
                            trace_fac
                            + tf.reshape(tf.reduce_sum((mu_i-mu_j)**2 / sig_i, axis=1), [N, 1]) 
                            - self.dim - det_fac
                            )

                with tf.name_scope("LossCal"):
                    self.energy_pos = energy(self.mu_embed, self.sig_embed, self.mu_embed_pos, self.sig_embed_pos) 
                    self.energy_neg = energy(self.mu_embed, self.sig_embed, self.mu_embed_neg, self.sig_embed_neg) 
                    self.loss = tf.reduce_mean(tf.maximum(0.0, self.Closs - self.energy_pos + self.energy_neg, name='MarginLoss'))

                self.train_step = getattr(tf.train, self.optimizer)(self.lr).minimize(self.loss)

                with tf.name_scope("clip_op"):
                    def clip_ops_graph():
                        def clip_val_ref(embedding, idxs):
                            with tf.name_scope("clip_val"):
                                to_update = tf.nn.embedding_lookup(embedding, idxs)
                                to_update = tf.maximum(lower_logsig, tf.minimum(upper_logsig, to_update))
                                return tf.scatter_update(embedding, idxs, to_update)

                        def clip_norm_ref(embedding, idxs):
                            with tf.name_scope("clip_norm_ref"):
                                to_update = tf.nn.embedding_lookup(embedding, idxs)
                                to_update = tf.clip_by_norm(to_update, self.norm_cap, axes=1)
                                return tf.scatter_update(embedding, idxs, to_update)

                        clip1 = clip_norm_ref(self.mu, self.u_id)
                        clip2 = clip_norm_ref(self.mu_out, self.v_pos_id)
                        clip3 = clip_norm_ref(self.mu_out, self.v_neg_id)
                        clip4 = clip_val_ref(self.logsig, self.u_id)
                        clip5 = clip_val_ref(self.logsig_out, self.v_pos_id)
                        clip6 = clip_val_ref(self.logsig_out, self.v_neg_id)
                        
                        return [clip1, clip2, clip3, clip4, clip5, clip6]

                    self.clip_op = clip_ops_graph()


    def train(self, get_batch):
        print ("[+] Start gaussian embedding ...")
        self.logger.info("[+] Start gaussian embedding ...")
        loss = 0.0
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i, batch in enumerate(get_batch()):
                input_dict = {self.u_id: batch[0],
                        self.v_pos_id: batch[1],
                        self.v_neg_id: batch[2]}
                self.train_step.run(input_dict)
                loss += self.loss.eval(input_dict)

                if self.normclip:
                    sess.run(self.clip_op, feed_dict=input_dict)
                
                if (i + 1) % self.show_num == 0:
                    print ("Epoch %d, Loss: %f" % (i+1, np.sum(loss / self.show_num)))
                    self.logger.info("Epoch %d, Loss: %f\n" % (i+1, np.sum(loss / self.show_num)))
                    loss = 0.0

            return np.array(sess.run(self.mu)), np.array(sess.run(self.logsig))

    def show_graph(self):
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            #print (sess.run(self.loss))
        writer.close()


if __name__ == "__main__":
    params = {}
    params["learning_rate"] = 0.01
    params["optimizer"] = "AdamOptimizer"
    params["num_nodes"] = 100
    params["embed_size"] = 32
    params["Closs"] = 1.0
    params["spherical"] = False 
    params["fixvar"] = False
    params["wout"] = False
    params["normclip"] = False
    params["show_num"] = 1000
    params["logger"] = "logger"
    params["batch_size"] = 100
    NE = NodeEmbedding(params)
    NE.build_graph()
    NE.show_graph()