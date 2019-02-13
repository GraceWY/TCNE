import os
import sys
import tensorflow as tf
import math
import random
import numpy as np

from utils import common_tools as ct
from utils.data_handler import DataHandler as dh

import pdb

INT=tf.int32
FLOAT = np.float64
DEBUG = False 

class TagConditionedEmbedding(object):
    """ Tag conditioned Network Embedding
    """
    def __init__(self, params, features):

        """ agg parameters
        """
        self.agg_neighbor_num = params["aggregator"]["agg_neighbor_num"]
        self.feature_num = params["aggregator"]["feature_num"]
        self.features = features

        """ entity embedding parameters
        """
        self.en_embed_size = params["en_embed_size"]
        self.en_num = params["en_num"]
        self.nce_k = params["generative_net"]["nce_k"]

        """ tag embedding parameters
        """
        self.tag_embed_size = params["tag_embed_size"]
        self.tag_num = params["tag_num"]
        self.spherical = params["tag_embedding"]["spherical"]
        self.tag_trainable = params["tag_embedding"]["tag_trainable"]

        self.tag_pre_train = ""
        if "tag_pre_train" in params["tag_embedding"]:
            self.tag_pre_train = params["tag_embedding"]["tag_pre_train"] # whether use pretrain parameters

        # The constant in margin-based loss
        self.Closs = 1.0 if "Closs" not in params["tag_embedding"] else params["tag_embedding"]["Closs"]
        # whether the w and c use the uniform parameters
        self.wout = False if "wout" not in params["tag_embedding"] else params["tag_embedding"]["wout"] 

        # whether to clip the range of mu and sigma for distribution
        self.normclip = False if "normclip" not in params["tag_embedding"] else params["tag_embedding"]["normclip"]
        self.varclip = False if "varclip" not in params["tag_embedding"] else params["tag_embedding"]["varclip"]

        """ lower_sig: element-wise lower bound for sigma
            upper_sig: element-wise upper bound for sigma
            norm_cap: upper bound of norm of mu
        """
        self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = 0.02, 5.0, 3.0, 1, 0.05
        if "lower_sig" in params["tag_embedding"]:
            self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = params["tag_embedding"]["lower_sig"], params["tag_embedding"]["upper_sig"], params["tag_embedding"]["norm_cap"], params["tag_embedding"]["mu_scale"], params["tag_embedding"]["var_scale"] 


        # model paramters
        self.output_embed_size = params["output_embed_size"]
        self.lr = params["learning_rate"]
        self.optimizer = params["optimizer"]
        self.batch_size = params["batch_size"]
        self._lambda = params["lambda"]
        self.show_num = params["show_num"] 
        self.logger = params["logger"]
        fn = "ckpt/TCNE_TagNum(%d)_TagEmbedSize(%d)_EnNum(%d)_EnEmbedSize(%d)_spherical(%r)" \
                % (self.tag_num, self.tag_embed_size, self.en_num, self.en_embed_size, self.spherical)
        self.model_save_path = os.path.join(params["res_home"], fn)

        self.tensor_graph = tf.Graph()
        self.build_graph()


    def build_graph(self):
        # parameter for Gaussian Embedding
        mu_scale = self.mu_scale * math.sqrt(3.0/(1.0 * self.tag_embed_size))
        logvar_scale = math.log(self.var_scale)
        lower_logsig = math.log(self.lower_sig)
        upper_logsig = math.log(self.upper_sig)

        with self.tensor_graph.as_default():
            tf.set_random_seed(random.randint(0, 1e9))

            def clip_by_min(tensor, _min=1e-3):
                return tf.clip_by_value(tensor, _min, float("inf"))
            

            with tf.name_scope("GaussEmbedding"):
                with tf.name_scope("Input"):
                    self.tag_placeholders = {
                        "u_id": tf.placeholder(name="u_id", dtype=INT, shape=[None]),
                        "p_id": tf.placeholder(name="p_id", dtype=INT, shape=[None]),
                        "n_id": tf.placeholder(name="n_id", dtype=INT, shape=[None])
                    }

                with tf.name_scope("Variable"):
                    self.mu = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                            -mu_scale, mu_scale, dtype = FLOAT), name="mu", dtype=FLOAT, trainable=self.tag_trainable)

                    if self.wout:
                        self.mu_out = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                -mu_scale, mu_scale, dtype = FLOAT), name="mu_out", dtype=FLOAT, trainable=self.tag_trainable)

                    if self.spherical:
                        self.logsig = tf.Variable(tf.random_uniform([self.tag_num, 1], \
                                logvar_scale, logvar_scale, dtype = FLOAT), name="logsigma", dtype=FLOAT, trainable=self.tag_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.tag_num, 1], \
                                    logvar_scale, logvar_scale, dtype = FLOAT), name="logsigma_out", dtype=FLOAT, trainable=self.tag_trainable)

                    else:
                        self.logsig = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                logvar_scale, logvar_scale, dtype = FLOAT), name="logsigma", dtype=FLOAT, trainable=self.tag_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                    logvar_scale, logvar_scale, dtype = FLOAT), name="logsigma_out", dtype=FLOAT, trainable=self.tag_trainable)

                    if not self.wout:
                        self.mu_out, self.logsig_out = self.mu, self.logsig

                with tf.name_scope("GetEmbedding"):
                    self.mu_embed = tf.nn.embedding_lookup(self.mu, self.tag_placeholders["u_id"])
                    self.mu_embed_pos = tf.nn.embedding_lookup(self.mu_out, self.tag_placeholders["p_id"]) 
                    self.mu_embed_neg = tf.nn.embedding_lookup(self.mu_out, self.tag_placeholders["n_id"]) 
                    self.sig_embed = tf.exp(tf.nn.embedding_lookup(self.logsig, self.tag_placeholders["u_id"]))
                    self.sig_embed_pos = tf.exp(tf.nn.embedding_lookup(self.logsig_out, self.tag_placeholders["p_id"]))
                    self.sig_embed_neg = tf.exp(tf.nn.embedding_lookup(self.logsig_out, self.tag_placeholders["n_id"]))

                def energy(mu_i, sig_i, mu_j, sig_j):
                    """E(P[i], P[j]) = Negative KL divergence
                    """
                    N = self.batch_size 
                    if self.spherical:
                        trace_fac = self.dim * clip_by_min(sig_j / sig_i)
                        det_fac = self.dim * tf.log(clip_by_min(sig_j / sig_i))
                    else:
                        trace_fac = tf.reshape(tf.reduce_sum(clip_by_min(sig_j / sig_i), axis=1), [N, 1]) 
                        det_fac = tf.reshape(tf.reduce_sum(tf.log(clip_by_min(sig_j)) - tf.log(clip_by_min(sig_i)), axis=1), [N, 1])

                    return -0.5 * (
                            trace_fac
                            + tf.reshape(tf.reduce_sum(clip_by_min((mu_i-mu_j)**2 / sig_i), axis=1), [N, 1]) 
                            - self.tag_embed_size - det_fac
                            )

                with tf.name_scope("LossCal"):
                    self.energy_pos = energy(self.mu_embed, self.sig_embed, self.mu_embed_pos, self.sig_embed_pos) 
                    self.energy_neg = energy(self.mu_embed, self.sig_embed, self.mu_embed_neg, self.sig_embed_neg) 
                    self.tag_loss = tf.reduce_mean(tf.maximum(FLOAT(0.0), self.Closs - self.energy_pos + self.energy_neg, name='MarginLoss'))

                with tf.name_scope("ClipOp"):
                    """ Clip variance
                    """
                    def clip_ops_graph_var():
                        def clip_var_ref(embedding, idxs):
                            with tf.name_scope("clip_var"):
                                to_update = tf.nn.embedding_lookup(embedding, idxs)
                                to_update = tf.maximum(FLOAT(lower_logsig), tf.minimum(FLOAT(upper_logsig), to_update))
                                return tf.scatter_update(embedding, idxs, to_update)

                        clip1 = clip_var_ref(self.logsig, self.tag_placeholders["u_id"])
                        clip2 = clip_var_ref(self.logsig_out, self.tag_placeholders["p_id"])
                        clip3 = clip_var_ref(self.logsig_out, self.tag_placeholders["n_id"])

                        return [clip1, clip2, clip3]

                    """ Clip mu
                    """
                    def clip_ops_graph_norm():
                        def clip_norm_ref(embedding, idxs):
                            with tf.name_scope("clip_norm_ref"):
                                to_update = tf.nn.embedding_lookup(embedding, idxs)
                                to_update = tf.clip_by_norm(to_update, self.norm_cap, axes=1)
                                return tf.scatter_update(embedding, idxs, to_update)

                        clip1 = clip_norm_ref(self.mu, self.tag_placeholders["u_id"])
                        clip2 = clip_norm_ref(self.mu_out, self.tag_placeholders["p_id"])
                        clip3 = clip_norm_ref(self.mu_out, self.tag_placeholders["n_id"])
                        
                        return [clip1, clip2, clip3]

                    self.clip_op_norm = clip_ops_graph_norm()
                    self.clip_op_var = clip_ops_graph_var()


            with tf.name_scope("EntityEmbedding"):
                with tf.name_scope("Input"):
                    self.entity_placeholders = {
                        "u_id": tf.placeholder(name="u_id", dtype=INT, shape=[None]), # center
                        "u_t": tf.placeholder(name="u_tag", dtype=FLOAT, shape=[None, self.tag_num]),
                        "u_noise": tf.placeholder(name="u_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size]),
                        "p_id": tf.placeholder(name="pos_id", dtype=INT, shape=[None]), # positive
                        "p_t": tf.placeholder(name="pos_tag", dtype=FLOAT, shape=[None, self.tag_num]),
                        "p_noise": tf.placeholder(name="pos_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size]),
                        "n_id": tf.placeholder(name="neg_id", dtype=INT, shape=[None]), # negative
                        "n_t": tf.placeholder(name="neg_tag", dtype=FLOAT, shape=[None, self.tag_num]),
                        "n_noise": tf.placeholder(name="neg_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size]),
                        "u_neighbors": tf.placeholder(name = "u_neighbors", dtype = INT, shape = [None, self.agg_neighbor_num]),
                        "p_neighbors": tf.placeholder(name = "p_neighbors", dtype = INT, shape = [None, self.agg_neighbor_num]),
                        "n_neighbors": tf.placeholder(name = "n_neighbors", dtype = INT, shape = [None, self.agg_neighbor_num]),
                    }
                with tf.name_scope("Aggregator"):
                    self.W_agg1 = tf.get_variable("W_agg1", [self.feature_num, self.en_embed_size], dtype = FLOAT,
                            initializer = tf.contrib.layers.xavier_initializer(dtype = tf.as_dtype(FLOAT)))
                    self.feature_mat = tf.constant(self.features, dtype = FLOAT, name = "feature_mat")
                    def AGG(en_ids, neighbors):
                        # batch_size * feature_num
                        u_emd = tf.nn.embedding_lookup(self.feature_mat, en_ids)
                        # batch_size * agg_neighbor_num * feature_num
                        neighbors_emd = tf.nn.embedding_lookup(self.feature_mat, neighbors)
                        neighbors_emd_reshape = tf.reshape(neighbors_emd, [-1, self.feature_num])
                        neighbors_agg = tf.matmul(neighbors_emd_reshape, self.W_agg1)
                        neighbors_agg_3d = tf.reshape(neighbors_agg, [-1, self.agg_neighbor_num, self.en_embed_size])
                        h1_pre = tf.reduce_sum(neighbors_agg_3d, axis=1)
                        h1 = tf.nn.leaky_relu(h1_pre, alpha=0.01)
                        return h1


                def INFER(_en_ids, _tag_mask, _tag_noise, _neighbors):
                    """ en_ids: (batch_size x k)  (# of negative sampling)
                        tag_mask: (batch_size x k) x tag_num
                        tag_noise: (batch_size x k) x tag_num x tag_embed_size

                        Return: batch_size x en_embed_size
                    """
                    en_ids = tf.reshape(_en_ids, [-1, 1])
                    tag_mask = tf.reshape(_tag_mask, [-1, self.tag_num])
                    tag_noise = tf.reshape(_tag_noise, [-1, self.tag_num, self.tag_embed_size])
                    neighbors = tf.reshape(_neighbors, [-1, self.agg_neighbor_num])
                    with tf.name_scope("Aggregator"):
                        en_X = AGG(en_ids, neighbors) # en_x : (batch_size x k) x en_embed_size
                        self.en_X = tf.reshape(en_X, [-1, self.en_embed_size])

                    with tf.name_scope("DynamicTagDist"):
                        with tf.variable_scope("DynTagDistVar", reuse = tf.AUTO_REUSE):
                            self.W_alpha = tf.get_variable("W_alpha", [self.en_embed_size, self.tag_num], dtype = FLOAT,
                                    initializer = tf.contrib.layers.xavier_initializer(dtype = tf.as_dtype(FLOAT)))

                        # add tricks to guarantee the stalibility of softmax
                        self.alpha_before_softmax = tf.matmul(en_X, self.W_alpha)   # (batch_sizexk) x tag_num
                        self.alpha_before_softmax_max = tf.stop_gradient(tf.reduce_max(self.alpha_before_softmax, axis = 1, keepdims = True))
                        self.alpha_exp = tag_mask * tf.exp(self.alpha_before_softmax - self.alpha_before_softmax_max)  # (batch_sizexk) x tag_num
                        self.alpha_sum = tf.reduce_sum(self.alpha_exp, axis=1, keepdims=True)
                        self.alpha_true_sum = tf.where(tf.less(tf.abs(self.alpha_sum), 1e-9), tf.ones_like(self.alpha_sum, dtype=FLOAT), \
                                self.alpha_sum)
                        self.alpha = tf.expand_dims(self.alpha_exp / self.alpha_true_sum, -1) # (batch_sizexk) x tag_num x 1

                        sig_std = tf.sqrt(clip_by_min(tf.exp(self.logsig)))

                        dist_sample = self.mu + sig_std * tag_noise # (batch_sizexk) x tag_num x tag_embed_size

                        tag_X = tf.reduce_sum(self.alpha * dist_sample, axis=1) # (batch_sizexk) x tag_embed_size

                    with tf.name_scope("GenerativeNet"):
                        with tf.variable_scope("GenNetVar", reuse=tf.AUTO_REUSE):
                            if DEBUG:
                                self.W_gen = tf.get_variable("W_gen", [self.tag_embed_size, self.output_embed_size], \
                                        dtype = FLOAT, initializer = tf.contrib.layers.xavier_initializer(dtype = tf.as_dtype(FLOAT))) 
                            else:
                                self.W_gen = tf.get_variable("W_gen", [self.tag_embed_size + self.en_embed_size, self.output_embed_size],\
                                        dtype = FLOAT, initializer = tf.contrib.layers.xavier_initializer(dtype = tf.as_dtype(FLOAT))) 
                        if DEBUG:
                            X = tag_X
                            # X = self.W_alpha 
                        else:
                            X = tf.concat([en_X, tag_X], 1)
                        # Y = tf.nn.leaky_relu(tf.matmul(X, self.W_gen), alpha=0.01, name='EmbeddingLayer')

                        if DEBUG:
                            Y = X
                        else:
                            Y = tf.nn.tanh(tf.matmul(X, self.W_gen), name='EmbeddingLayer')
                        # Y = tf.nn.sigmoid(tf.matmul(X, self.W_gen), name='EmbeddingLayer')

                    return Y

                self.u_y = INFER(self.entity_placeholders["u_id"], self.entity_placeholders["u_t"], self.entity_placeholders["u_noise"], self.entity_placeholders["u_neighbors"]) # batch_size x output_embed_size
                self.p_y = INFER(self.entity_placeholders["p_id"], self.entity_placeholders["p_t"], self.entity_placeholders["p_noise"], self.entity_placeholders["p_neighbors"]) # batch_size x output_embed_size
                self._n_y = INFER(self.entity_placeholders["n_id"], self.entity_placeholders["n_t"], self.entity_placeholders["n_noise"], self.entity_placeholders["n_neighbors"])  # (batch_size x k) x output_embed_size
                self.n_y = tf.reshape(self._n_y, [-1, self.nce_k, self.output_embed_size]) # batch_size x k x output_embed_size
                #self.n_y = tf.where(tf.is_nan(self.n_y_with_nan), tf.zeros_like(self.n_y_with_nan)+1e-6, self.n_y_with_nan) 


                with tf.name_scope("NCELoss"):
                    u_y_3d = tf.reshape(self.u_y, [-1, 1, self.output_embed_size])

                    # dim: batch_size x k
                    # neg_dot = tf.squeeze(tf.matmul(u_y_3d, self.n_y, transpose_b=True))
                    neg_dot = tf.reduce_sum(tf.matmul(u_y_3d, self.n_y, transpose_b=True), 1)
                    self.neg_dot = neg_dot
                    self.pos = -tf.log(clip_by_min(tf.sigmoid(tf.reduce_sum(self.u_y * self.p_y, axis=1))))
                    self.neg = tf.reduce_mean(tf.log(clip_by_min(tf.sigmoid(-neg_dot))), -1)
                    self.nce_loss = tf.reduce_mean(self.pos-self.neg)
                    self.margin_loss = tf.reduce_mean(tf.maximum(FLOAT(0.0), self.Closs - self.pos + self.neg))
                    #self.nce_loss = tf.reduce_mean(self.pos)

            self.loss = self.nce_loss + self._lambda * self.tag_loss
            # self.loss = self.nce_loss
            # self.loss = self.margin_loss
            self.train_step = getattr(tf.train, self.optimizer)(self.lr).minimize(self.loss)
            self.grad_Wgen = tf.gradients(self.loss, self.W_gen)
            self.grad_n_y = tf.gradients(self.loss, self.n_y)
            self.grad_u_y = tf.gradients(self.loss, self.u_y)
            self.grad_p_y = tf.gradients(self.loss, self.p_y)

            # Add ops to save and restore all the variables
            self.tag_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GaussEmbedding/Variable"))
            self.init_op = tf.global_variables_initializer()
            # self.sess = tf.Session(graph = self.tensor_graph)
            self.model_saver = tf.train.Saver()



    def train(self, get_batch):
        print ("[+] start node embedding ...")
        self.logger.info("[+] start node embedding ...\n")
        loss = 0.0

        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(self.init_op)

            """ Init the parameters of tag distribution
            """
            if len(self.tag_pre_train) != 0 and not DEBUG:
                print ("[+] reload pre train parameters of tag distribution from %s" % (self.tag_pre_train))
                self.logger.info("[+] save pre train parameters of tag distribution from %s\n" % (self.tag_pre_train))
                print (sess.run(self.mu)[0, :])
                self.tag_saver.restore(sess, self.tag_pre_train)
                print (sess.run(self.mu)[0, :])

            if DEBUG:
                print ("######################## mu\n")
                print (sess.run(self.mu))
                print ("######################## logsig\n")
                print (sess.run(self.logsig))
                pdb.set_trace()

            for i, batch in enumerate(get_batch()):
                tag_input_dict = {
                    self.tag_placeholders["u_id"]: batch["tag_u"],
                    self.tag_placeholders["p_id"]: batch["tag_v"],
                    self.tag_placeholders["n_id"]: batch["tag_n"]
                }
                en_input_dict = {
                    self.entity_placeholders["u_id"]: batch["en_u"],
                    self.entity_placeholders["p_id"]: batch["en_v"],
                    self.entity_placeholders["n_id"]: batch["en_n"],
                    self.entity_placeholders["u_t"]: batch["en_u_mask"],
                    self.entity_placeholders["p_t"]: batch["en_v_mask"],
                    self.entity_placeholders["n_t"]: batch["en_n_mask"],
                    self.entity_placeholders["u_noise"]: batch["en_u_noise"],
                    self.entity_placeholders["p_noise"]: batch["en_v_noise"],
                    self.entity_placeholders["n_noise"]: batch["en_n_noise"],
                    self.entity_placeholders["u_neighbors"]: batch["en_u_neighbors"],
                    self.entity_placeholders["p_neighbors"]: batch["en_v_neighbors"],
                    self.entity_placeholders["n_neighbors"]: batch["en_n_neighbors"]
                }

                # print ("batch: ", batch)
                # pdb.set_trace()

                input_dict = {}
                for k, v in tag_input_dict.items():
                    input_dict[k] = v
                for k, v in en_input_dict.items():
                    input_dict[k] = v

                
                if DEBUG:
                    print ("Loss, gradient before")
                    print ("show u_y")
                    print (batch["en_u"])
                    print (sess.run(self.u_y, feed_dict=input_dict))

                    print ("show p_y")
                    print (batch["en_v"])
                    print (sess.run(self.p_y, feed_dict=input_dict))
                    # print (sess.run(self.neg_dot, feed_dict=input_dict))
                    # print (sess.run(self.neg, feed_dict=input_dict))
                    # print ("show loss")
                    # print (sess.run(self.loss, feed_dict=input_dict))

                    print ("show n_y")
                    print (batch["en_n"])
                    print (sess.run(self.n_y, feed_dict=input_dict))

                    # print ("show W_alpha")
                    # print (sess.run(self.W_alpha))

                    print ("show pos and neg")
                    print (sess.run(self.pos, feed_dict=input_dict))
                    print (sess.run(self.neg, feed_dict=input_dict))

                    print ("Grad")
                    # print (("W_gen Grad: ", sess.run(self.grad_Wgen, feed_dict=input_dict)))
                    print (sess.run(self.grad_u_y, feed_dict=input_dict))
                    print (sess.run(self.grad_p_y, feed_dict=input_dict))
                    print (sess.run(self.grad_n_y, feed_dict=input_dict))
                    

                # self.train_step.run(input_dict)
                sess.run(self.train_step, feed_dict = input_dict)
                loss += sess.run(self.loss, feed_dict = input_dict)
                if DEBUG:
                    print ("After update grad:")
                    print (sess.run(self.pos, feed_dict=input_dict))
                    print (sess.run(self.neg, feed_dict=input_dict))
                    print (sess.run(self.loss, feed_dict=input_dict))
                

                # print (sess.run(self.nce_loss, feed_dict=input_dict))
                # print (sess.run(self.tag_loss, feed_dict=input_dict))
                if DEBUG:
                   pdb.set_trace()

                # clip mu
                if self.normclip:
                    sess.run(self.clip_op_norm, feed_dict=input_dict)

                # clip var
                if self.varclip:
                    sess.run(self.clip_op_var, feed_dict=input_dict)

                if  (i+1) % self.show_num == 0:
                    print ("Epoch %d, Loss: %f" % (i+1, np.sum(loss / self.show_num)))
                    self.logger.info("Epoch %d, Loss: %f" % (i+1, np.sum(loss / self.show_num)))
                    loss = 0.0

                    # save model
                    self.model_saver.save(sess, self.model_save_path, global_step=i+1)
                    self.model_saver.save(sess, self.model_save_path)

        return self.model_save_path


    def infer(self, inputs, model_path=None):
        print ("[+] start infer node embedding ...")
        self.logger.info("[+] start infer node embedding ...\n")

        input_dict = {
                self.entity_placeholders["u_id"]: inputs["u"],
                self.entity_placeholders["u_t"]: inputs["u_mask"],
                self.entity_placeholders["u_noise"]: inputs["u_noise"],
                self.entity_placeholders["u_neighbors"]: inputs["u_neighbors"]
        }
        pdb.set_trace()

        with tf.Session(graph=self.tensor_graph) as sess:
            self.model_saver.restore(sess, model_path)
            return sess.run(self.u_y, feed_dict=input_dict)
