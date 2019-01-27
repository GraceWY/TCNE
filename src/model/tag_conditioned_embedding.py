import os
import sys
import tensorflow as tf

INT=tf.int32
FLOAT=tf.float32

class Aggregator(object):
    """ AGG network
    """
    def __init__(self, placeholders, features, adj):
        self.dropout = dropout 
        self.bias = bias
        
        if neigh_input_dim is None:
            neigh_input_dim = input_dim


    def build_graph(tensor_graph, inputs):
        self.vars["neigh_"]


class TagConditionedEmbedding(object):
    """ Tag conditioned Network Embedding
    """
    def __init__(self, params):

        """ entity embedding parameters
        """
        self.en_embed_size = params["en_embed_size"]
        self.en_num = params["en_num"]
        self.double_layer = params["aggregator"]["double_layer"]
        self.nec_k = params["generative_net"]["nce_k"]

        """ tag embedding parameters
        """
        self.tag_embed_size = params["tag_embed_size"]
        self.tag_num = params["tag_num"] 
        self.spherical = params["spherical"]
        self.tag_trainable = params["tag_trainable"]

        self.tag_pre_train = ""
        if "tag_pre_train" in params:
            self.tag_pre_train = params["tag_pre_train"] # whether use pretrain parameters

        # The constant in margin-based loss
        self.Closs = 1.0 if "Closs" not in params else params["Closs"]
        # whether the w and c use the uniform parameters
        self.wout = False if "wout" not in params else params["wout"] 

        # whether to clip the range of mu and sigma for distribution
        self.normclip = False if "normclip" not in params else params["normclip"]
        self.varclip = False if "varclip" not in params else params["varclip"]

        """ lower_sig: element-wise lower bound for sigma
            upper_sig: element-wise upper bound for sigma
            norm_cap: upper bound of norm of mu
        """
        self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = 0.02, 5.0, 3.0, 1, 0.05
        if "lower_sig" in params:
            self.lower_sig, self.upper_sig, self.norm_cap, self.mu_scale, self.var_scale = params["lower_sig"], params["upper_sig"], params["norm_cap"], params["mu_scale"], params["var_scale"] 


        # model paramters
        self.lr = params["learning_rate"]
        self.optimizer = params["optimizer"]
        self.batch_size = params["batch_size"]
        self.lambda = params["lambda"]
        self.show_num = params["show_num"] 
        fn = "ckpt/TCNE_TagNum($d)_TagEmbedSize(%d)_EnNum(%d)_EnEmbedSize(%d)_spherical(%r)" \
                % (self.tag_num, self.tag_embed_size, self.en_num, self.en_embed_size, self.spherical)
        self.model_save_path = os.path.join(params["res_home"], fn)


    def build_graph(self):

        # parameter for Gaussian Embedding
        mu_scale = self.mu_scale * math.sqrt(3.0/(1.0 * self.tag_embed_size))
        logvar_scale = math.log(self.var_scale)
        lower_logsig = math.log(self.lower_sig)
        upper_logsig = math.log(self.upper_sig)

        with self.tensor_graph.as_default():
            tf.set_random_seed(random.randint(0, 1e9))
            
            # Add ops to save and restore all the variables
            tag_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GaussEmbedding/Variable"))
            
            model_saver = tf.train.Saver()

            with tf.name_scope("GaussEmbedding"):
                with tf.name_scope("Input"):
                    self.tag_placeholders = {
                        "u_id": tf.placeholder(name="u_id", dtype=INT, shape=[None]),
                        "p_id": tf.placeholder(name="p_id", dtype=INT, shape=[None]),
                        "n_id": tf.placeholder(name="n_id", dtype=INT, shape=[None])
                    }

                with tf.name_scope("Variable"):
                    self.mu = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                            -mu_scale, mu_scale), name="mu", dtype=FLOAT, trainable=self.tag_trainable)

                    if self.wout:
                        self.mu_out = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                -mu_scale, mu_scale), name="mu_out", dtype=FLOAT, trainable=self.tag_trainable)

                    if self.spherical:
                        self.logsig = tf.Variable(tf.random_uniform([self.tag_num, 1], \
                                logvar_scale, logvar_scale), name="logsigma", dtype=FLOAT, trainable=self.tag_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.tag_num, 1], \
                                    logvar_scale, logvar_scale), name="logsigma_out", dtype=FLOAT, trainable=self.tag_trainable)

                    else:
                        self.logsig = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                logvar_scale, logvar_scale), name="logsigma", dtype=FLOAT, trainable=self.tag_trainable)

                        if self.wout:
                            self.logsig_out = tf.Variable(tf.random_uniform([self.tag_num, self.tag_embed_size], \
                                    logvar_scale, logvar_scale), name="logsigma_out", dtype=FLOAT, trainable=self.tag_trainable)

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
                    self.tag_loss = tf.reduce_mean(tf.maximum(0.0, self.Closs - self.energy_pos + self.energy_neg, name='MarginLoss'))

                with tf.name_scope("ClipOp"):
                    """ Clip variance
                    """
                    def clip_ops_graph_var():
                        def clip_var_ref(embedding, idxs):
                            with tf.name_scope("clip_var"):
                                to_update = tf.nn.embedding_lookup(embedding, idxs)
                                to_update = tf.maximum(lower_logsig, tf.minimum(upper_logsig, to_update))
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
                        "u_t": tf.placeholder(name="u_tag", dtype=INT, shape=[None, self.tag_num]),
                        "u_noise": tf.placeholder(name="u_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size]),
                        "p_id": tf.placeholder(name="pos_id", dtype=INT, shape=[None]), # positive
                        "p_t": tf.placeholder(name="pos_tag", dtype=INT, shape=[None, self.tag_num]),
                        "p_noise": tf.placeholder(name="pos_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size]),
                        "n_id": tf.placeholder(name="neg_id", dtype=INT, shape=[None]), # negative
                        "n_t": tf.placeholder(name="neg_tag", dtype=INT, shape=[None, self.tag_num]),
                        "n_noise": tf.placeholder(name="neg_noise", dtype=FLOAT, shape=[None, self.tag_num, self.tag_embed_size])
                    }

                def INFER(_en_ids, _tag_mask, _tag_noise):
                    """ en_ids: (batch_size x k)  (# of negative sampling)
                        tag_mask: (batch_size x k) x tag_num
                        tag_noise: (batch_size x k) x tag_num x tag_embed_size

                        Return: batch_size x en_embed_size
                    """
                    en_ids = tf.reshape(_en_ids, [-1, 1])
                    tag_mask = tf.reshape(_tag_mask, [-1, self.tag_num])
                    tag_noise = tf.reshape(_tag_noise, [-1, self.tag_num, self.tag_embed_size])
                    with tf.name_scope("Aggregator"):
                        en_X = AGG(en_ids) # en_x : (batch_size x k) x en_embed_size

                    with tf.name_scope("DynamicTagDist"):
                        self.W_alpha = ct.glorot_init([self.en_embed_size, self.tag_num], FLOAT, name="W_alpha") 
                        tmp = tag_mask * tf.exp(tf.matmul(en_X, self.W_alpha))  # (batch_sizexk) x tag_num
                        self.alpha = tf.expand_dims(tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True), -1) # (batch_sizexk) x tag_num x 1

                        sig_std = tf.sqrt(tf.exp(self.logsig))

                        dist_sample = self.mu + sig_std * tag_noise # (batch_sizexk) x tag_num x tag_embed_size

                        tag_X = tf.reduce_sum(self.alpha * dist_sample, axis=1) # (batch_sizexk) x tag_embed_size

                    with tf.name_scope("GenerativeNet"):
                        X = tf.concat([en_X, tag_X], 1)
                        Y = tf.nn.leaky_relu(X, alpha=0.01, name='EmbeddingLayer')

                    return Y

                self.u_y = INFER(self.placeholders["u_id"], self.placeholders["u_t"], self.placeholders["u_noise"]) # batch_size x en_embed_size
                self.p_y = INFER(self.placeholders["p_id"], self.placeholders["p_t"], self.placeholders["p_noise"]) # batch_size x en_embed_size
                _n_y = INFER(self.placeholders["n_id"], self.placeholders["n_t"], self.placeholders["n_noise"]) # (batch_size x k) x en_embed_size 
                self.n_y = tf.reshape(_n_y, [self.batch_size, self.nec_k, self.en_embed_size]) # batch_size x k x en_embed_size


                with tf.name_scope("NCELoss"):
                    u_y_3d = tf.reshape(self.u_y, [-1, 1, self.en_embed_size])

                    # dim: batch_size x k
                    neg_dot = tf.squeeze(tf.matmul(u_y_3d, self.n_y, transpose_b=True))
                    self.nce_loss = tf.reduce_mean(-tf.log(tf.sigmoid(tf.reduce_sum(self.u_y * self.p_y, axis=1)))-tf.log(tf.sigmoid(-neg_dot)))

            self.loss = self.nec_loss + self.lambda * self.tag_loss
            self.train_step = getattr(tf.train, self.optimizer)(self.lr).minimize(self.loss)


    def train(self, get_batch):
        print ("[+] start node embedding ...")
        self.logger.info("[+] start node embedding ...\n")
        loss = 0.0
        with tf.Session(graph=self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())

            """ Init the parameters of tag distribution
            """
            if len(self.tag_pre_train) != 0:
                print ("[+] reload pre train parameters of tag distribution from %s" % (self.tag_pre_train))
                self.logger.info("[+] save pre train parameters of tag distribution from %s\n" % (self.tag_pre_train))
                tag_saver.restore(sess, self.tag_pre_train)


            for i, batch in enumerate(get_batch()):
                tag_input_dict = {
                    self.tag_placeholders["u_id"]: batch["t_u_id"],
                    self.tag_placeholders["p_id"]: batch["t_p_id"],
                    self.tag_placeholders["n_id"]: batch["t_n_id"]
                }
                en_input_dict = {
                    self.entity_placeholders["u_id"]: batch["e_u_id"],
                    self.entity_placeholders["p_id"]: batch["e_p_id"],
                    self.entity_placeholders["n_id"]: batch["e_n_id"],
                    self.entity_placeholders["u_t"]: batch["e_u_t"],
                    self.entity_placeholders["p_t"]: batch["e_p_t"],
                    self.entity_placeholders["n_t"]: batch["e_n_t"],
                    self.entity_placeholders["u_noise"]: batch["e_u_noise"],
                    self.entity_placeholders["p_noise"]: batch["p_u_noise"],
                    self.entity_placeholders["n_noise"]: batch["n_u_noise"]
                }

                input_dict = {}
                for k, v in tag_input_dict.items():
                    input_dict[k] = v
                for k, v in en_input_dict.items():
                    input_dict[k] = v
                self.train_step.run(input_dict)
                loss += self.loss.eval(input_dict)

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
                    model_saver.save(sess, self.model_save_path, global_step=i+1)

            return self.model_save_path

