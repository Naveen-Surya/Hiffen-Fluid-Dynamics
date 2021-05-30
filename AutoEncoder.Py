import tensorflow as tf

def Encoder(Y, n_hidden_layers, n_z, prob):
    w_initialiser = tf.contrib.layers.xavier_initializer()
    input_shape = Y.get_shape()

    with tf.variable_scope("Encod_hid1", reuse = tf.AUTO_REUSE):
        w1 = tf.get_variable("w1", shape = [input_shape[1], n_hidden_layers], initializer = w_initialiser)
        b1 = tf.get_variable("b1", shape = [n_hidden_layers], initializer = tf.constant_initializer(0.))
        h1 = tf.matmul(Y,w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, prob)

    with tf.variable_scope("Encod_hid2", reuse = tf.AUTO_REUSE):
        w2 = tf.get_variable("w2", shape = [n_hidden_layers,n_hidden_layers], initializer = w_initialiser)
        b2 = tf.get_variable("b2", shape = [n_hidden_layers], initializer = tf.constant_initializer(0.))
        h2 = tf.matmul(h1,w2) + b2
        h2 = tf.nn.sigmoid(h2)
        h2 = tf.nn.dropout(h2,prob)

    with tf.variable_scope("Encod_hidz", reuse = tf.AUTO_REUSE):
        w3 = tf.get_variable("w3", shape = [n_hidden_layers, n_z*2], initializer = w_initialiser)
        b3 = tf.get_variable("b3", shape = [n_z*2], initializer = tf.constant_initializer(0.))
        h3 = tf.matmul(h2,w3) + b3
        mean = h3[:, : n_z]
        std = tf.nn.softplus(h3[:, n_z :]) + 1e-6

        return mean, std

def Decoder(z, n_hidden_layers, n_out ,prob):
    w_initialiser = tf.contrib.layers.xavier_initializer()
    z_shape = z.get_shape()

    with tf.variable_scope("decod_hid1", reuse = tf.AUTO_REUSE):
        w4 = tf.get_variable("w4", shape = [z_shape[1],n_hidden_layers], initializer = w_initialiser)
        b4 = tf.get_variable("b4", shape = [n_hidden_layers], initializer = tf.constant_initializer(0.))
        h4 = tf.matmul(z,w4) + b4
        h4 = tf.nn.elu(h4)
        h4 = tf.nn.dropout(h4,prob)

    with tf.variable_scope("decod_hid2", reuse = tf.AUTO_REUSE):
        w5 = tf.get_variable("w5", shape = [n_hidden_layers, n_hidden_layers], initializer = w_initialiser)
        b5 = tf.get_variable("b5", shape = [n_hidden_layers], initializer = tf.constant_initializer(0.))
        h5 = tf.matmul(h4,w5) + b5
        h5 = tf.nn.elu(h5)
        h5 = tf.nn.dropout(h5, prob)

    with tf.variable_scope("decoder_out", reuse = tf.AUTO_REUSE):
        w6 = tf.get_variable("w6",shape = [n_hidden, n_out], initializer = w_initialiser)
        b6 = tf.get_variable("b6", shape = [n_out], initializer = tf.constant_initializer(0.))
        h6 = tf.matmul(h5,w6) + b6
        h6 = tf.nn.sigmoid(h6)

        return h6

def VAE(Y,n_hiddEncode,n_z, n_hiddenDecod, prob ):
    Y_shape = Y.get_shape()
    n_output = Y_shape[1]

    mean, std = Encoder(Y,n_hiddEncode, n_z,prob)

    z = mean + std*tf.random_normal(tf.shape(mean,out_type = tf.int32), 0, 1, dtype = tf.float32)

    Y_out = Decoder(z,n_hiddenDecod,n_output,prob)
    Y_out = tf.clip_by_value(Y_out,1e-8, 1 - 1e-8)

    likelihood = tf.reduce_mean(tf.reduce_sum(Y*tf.log(Y_out) + (1-Y)*tf.log(1- Y_out),1))
    KLD= tf.reduce_mean(0.5*tf.reduce_sum(1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), 1))
    Reconstrcution_error = -1*likelihood
    Regularization_error = KLD
    ELBO = Reconstrcution_error + Regularization_error
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(-ELBO)

    return  z ,Y_out, Recon_error, Regularization_error, ELBO
