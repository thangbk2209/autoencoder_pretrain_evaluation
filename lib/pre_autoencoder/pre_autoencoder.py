import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn
import time

from config import *
from lib.preprocess_data.preprocessing_data import DataPreprocessor

matplotlib.use(Config.PLT_ENV)


class PreAutoEncoder:
    def __init__(self, data=None, scaler=None, train_size=None, valid_size=None, sliding_encoder=None, batch_size=None,
                 num_units_lstm=None, dropout_rate=None, variation_dropout=False, activation=None, optimizer=None,
                 variant=None, optimizer_approach=None, learning_rate=None, epochs=None, patience=None,
                 num_particle=None, w_old_velocity=None, w_local_best_position=None, w_global_best_position=None):
        self.data = data
        self.scaler = scaler
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.batch_size = batch_size
        self.num_units_lstm = num_units_lstm
        self.dropout_rate = dropout_rate
        self.variation_dropout = variation_dropout
        self.optimizer_approach = optimizer_approach
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.num_particle = num_particle

        self.variant = variant
        self.activation = activation
        if self.activation == 'sigmoid':
            self.activation_func = tf.nn.sigmoid
        elif self.activation == 'relu':
            self.activation_func = tf.nn.relu
        elif self.activation == 'tanh':
            self.activation_func = tf.nn.tanh
        elif self.activation == 'elu':
            self.activation_func = tf.nn.elu
        else:
            print(">>> Can not apply your activation <<<")

        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.optimizer_method = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'adam':
            self.optimizer_method = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.optimizer_method = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            print(">>> Can not apply your optimizer <<<")
        self.sess = None
        self.graph = None
        self.create_name()
        if Config.DATA_EXPERIMENT == 'google_trace':
            x_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type']
            y_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['predict_data']

        self.evaluation_path = Config.EVALUATION_PATH.format('bnn/autoencoder', self.optimizer_approach, x_data_name, 
                                                             y_data_name)
        self.model_saved_path = Config.MODEL_SAVE_PATH.format('bnn/autoencoder', self.optimizer_approach, x_data_name,
                                                              y_data_name) + self.file_name + '/model'
        self.train_loss_path = Config.TRAIN_LOSS_PATH.format('bnn/autoencoder', self.optimizer_approach,
                                                             x_data_name, y_data_name) + self.file_name + '.png'

        self.preprocessing_data()
        self.load_model()

    def create_name(self):
        def create_name_network(num_units):
            name = ''
            for i in range(len(num_units)):

                if (i == len(num_units) - 1):
                    name += str(num_units[i])
                else:
                    name += str(num_units[i]) + '_'
            return name

        name_lstm = create_name_network(self.num_units_lstm)
        if self.variant:
            part1 = 'sli_encoder-{}_batch-{}_name_lstm-{}'.format(self.sliding_encoder, self.batch_size, name_lstm)
        else:
            part1 = 'variant_sli_encoder-{}_batch-{}_name_lstm-{}'\
                .format(self.sliding_encoder, self.batch_size, name_lstm)
        part2 = 'activation-{}_optimizer_{}'.format(self.activation, self.optimizer)
        part4 = '_num_par-{}'.format(self.num_particle)
        if self.variation_dropout:
            part3 = '_dropout_rate-{}'.format(self.dropout_rate)
        else:
            part3 = ''

        self.file_name = part1 + part2 + part3 + part4

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train_encoder, self.x_train_decoder, self.y_train_decoder, self.x_valid_encoder, self.x_valid_decoder,\
            self.y_valid_decoder, self.x_test_encoder, self.x_test_decoder, self.y_test_decoder\
            = data_preprocessor.init_data_autoencoder(self.sliding_encoder)

    def load_model(self):
        metadata = '{}.meta'.format(self.model_saved_path)
        model_graph_dir = self.model_saved_path.rsplit(os.sep, 1)[0]
        print(model_graph_dir)
        if not os.path.isfile(metadata):
            print('[ERROR] Not found autoencoder model in path: {}'.format(self.model_saved_path))
            self.fit()

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            checkpoint = tf.train.import_meta_graph(metadata)
            checkpoint.restore(self.sess, tf.train.latest_checkpoint(model_graph_dir))
            self.hidden_state_encoder = self.graph.get_tensor_by_name('encoder/hidden_state_encoder:0')
            self.x_encoder = self.graph.get_tensor_by_name('x_encoder:0')
            self.x_decoder = self.graph.get_tensor_by_name('x_decoder:0')
            self.y_decoder = self.graph.get_tensor_by_name('y_decoder:0')

    def draw_train_loss(self, cost_train_set, cost_valid_set):
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.train_loss_path)
        plt.close()
    
    

    def compute_state_vector(self):

        state_train = self.sess.run(self.hidden_state_encoder, feed_dict={self.x_encoder: self.x_train_encoder})
        state_valid = self.sess.run(self.hidden_state_encoder, feed_dict={self.x_encoder: self.x_valid_encoder})
        state_test = self.sess.run(self.hidden_state_encoder, feed_dict={self.x_encoder: self.x_test_encoder})

        return state_train, state_valid, state_test

    def init_RNN(self, num_units):
        num_layers = len(num_units)
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                cell = tf.nn.rnn_cell.LSTMCell(num_units[i], activation=self.activation_func)
                if self.variation_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                                                         state_keep_prob=self.dropout_rate, variational_recurrent=True,
                                                         input_size=self.x_train.shape[2], dtype=tf.float32)
                hidden_layers.append(cell)
            else:
                cell = tf.nn.rnn_cell.LSTMCell(num_units[i], activation=self.activation_func)
                if self.variation_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                                                         state_keep_prob=self.dropout_rate, variational_recurrent=True,
                                                         input_size=num_units[i - 1], dtype=tf.float32)
                hidden_layers.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)

    def early_stopping(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array) - patience:]
        check = 0
        for val in arr:
            if val >= value:
                check += 1
        if check >= patience - 1:
            return True
        else:
            return False

    def __fit_with_bp(self):
        tf.reset_default_graph()

        # define placeholder of graph
        x_encoder = tf.placeholder("float", [None, self.x_train_encoder.shape[1], self.x_train_encoder.shape[2]],
                                   name='x_encoder')
        x_decoder = tf.placeholder("float", [None, self.x_train_decoder.shape[1], self.x_train_decoder.shape[2]],
                                   name='x_decoder')
        y_decoder = tf.placeholder("float", [None, self.y_train_decoder.shape[1], self.y_train_decoder.shape[2]],
                                   name='y_decoder')

        with tf.variable_scope('encoder'):
            encoder_cell = self.init_RNN(self.num_units_lstm)
            outputs_encoder, state_encoder = tf.nn.dynamic_rnn(encoder_cell, x_encoder, dtype="float32")
            outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
            if self.variant:
                mean = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_mean')
                log_var = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_log_var')
                batch = tf.shape(mean)[0]
                dim = tf.keras.backend.int_shape(mean)[1]
                # by default, random_normal has mean=0 and std=1.0
                epsilon = tf.random_normal(shape=(batch, dim))
                hidden_state_encoder = mean + tf.exp(0.5 * log_var) * epsilon
            else:
                hidden_state_encoder = state_encoder[-1].h
            hidden_state_encoder = tf.identity(hidden_state_encoder, name='hidden_state_encoder')
        with tf.variable_scope('decoder'):
            decoder_cell = self.init_RNN(self.num_units_lstm)
            outputs_decoder, state_decoder = tf.nn.dynamic_rnn(decoder_cell, x_decoder, dtype="float32",
                                                               initial_state=state_encoder)
            outputs_decoder = tf.identity(outputs_decoder, name='outputs_decoder')
            prediction = outputs_decoder[:, :, -1]
            prediction = tf.reshape(prediction, (tf.shape(outputs_decoder)[0], 1, tf.shape(outputs_decoder)[1]))
        prediction = tf.identity(prediction, name='prediction')
        # loss_function
        if self.variant:
            # Kullback-Leibler divergence with 2 gaussian distribution when reparameterize is:
            # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1 -mu2)^2)/2*sigma2^2 - 1/2
            # => just for one dimension, we need to scale it up
            reconstruction_loss = tf.reduce_mean(tf.square(y_decoder - prediction))
            kl = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(2.0 * log_var) - 2.0 * log_var - 1.0)
            loss = reconstruction_loss + kl
        else:
            loss = tf.reduce_mean(tf.square(y_decoder - prediction))
        loss = tf.identity(loss, name='loss')
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        cost_train_set = []
        cost_valid_set = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            print("start training autoencoder model")
            for epoch in range(self.epochs):
                # Train with each example
                start_time = time.time()
                total_batch = int(len(self.x_train_encoder) / self.batch_size)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder = self.x_train_encoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_xs_decoder = self.x_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys = self.y_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder,
                                                   y_decoder: batch_ys})

                    avg_cost += sess.run(loss, feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder,
                                                          y_decoder: batch_ys}) / total_batch
                # Display logs per epoch step
                training_history = 'Epoch autoencoder %04d: cost = %.9f with time: %.9f'\
                    % (epoch + 1, avg_cost, time.time() - start_time)
                print(training_history)
                cost_train_set.append(avg_cost)
                val_cost = sess.run(loss, feed_dict={x_encoder: self.x_valid_encoder, x_decoder: self.x_valid_decoder,
                                                     y_decoder: self.y_valid_decoder})
                cost_valid_set.append(val_cost)
                if epoch > self.patience:
                    if self.early_stopping(cost_train_set, self.patience):
                        print('early stop training auto encoder model')
                        break

            state_train = sess.run(hidden_state_encoder, feed_dict={x_encoder: self.x_train_encoder})
            state_valid = sess.run(hidden_state_encoder, feed_dict={x_encoder: self.x_valid_encoder})
            state_test = sess.run(hidden_state_encoder, feed_dict={x_encoder: self.x_test_encoder})

            test_loss = sess.run(loss, feed_dict={x_encoder: self.x_test_encoder, x_decoder: self.x_test_decoder,
                                                  y_decoder: self.y_test_decoder})

            print('Saving model to storage')
            saver.save(sess, self.model_saved_path)
            self.draw_train_loss(cost_train_set, cost_valid_set)
            with open(self.evaluation_path, 'a+') as f:
                f.write(self.file_name + ',' + str(test_loss) + '\n')
            print(' === Training autoencoder with back propagation complete complete ===')

    def __fit_with_pso(self):
        space = SpaceAutoEncoder(self.num_particle, self.x_train_encoder, self.x_train_decoder, self.y_train_decoder,
                                 self.x_valid_encoder, self.x_valid_decoder, self.y_valid_decoder, self.x_test_encoder,
                                 self.x_test_decoder, self.y_test_decoder, self.batch_size, self.epochs)

        space.particles = []
        for _ in range(space.num_particle):
            tf.reset_default_graph()

            # define placeholder of graph
            x_encoder = tf.placeholder("float", [None, self.x_train_encoder.shape[1], self.x_train_encoder.shape[2]],
                                       name='x_encoder')
            x_decoder = tf.placeholder("float", [None, self.x_train_decoder.shape[1], self.x_train_decoder.shape[2]],
                                       name='x_decoder')
            y_decoder = tf.placeholder("float", [None, self.y_train_decoder.shape[1], self.y_train_decoder.shape[2]],
                                       name='y_decoder')

            with tf.variable_scope('encoder'):
                encoder_cell = self.init_RNN(self.num_units_lstm)
                outputs_encoder, state_encoder = tf.nn.dynamic_rnn(encoder_cell, x_encoder, dtype="float32")
                outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
                if self.variant:
                    print('using variant for auto encoder!!!')
                    mean = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_mean')
                    log_var = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_log_var')
                    batch = tf.shape(mean)[0]
                    dim = tf.keras.backend.int_shape(mean)[1]
                    # by default, random_normal has mean=0 and std=1.0
                    epsilon = tf.random_normal(shape=(batch, dim))
                    hidden_state_encoder = mean + tf.exp(0.5 * log_var) * epsilon
                else:
                    hidden_state_encoder = state_encoder[-1].h
                hidden_state_encoder = tf.identity(hidden_state_encoder, name='hidden_state_encoder')
            with tf.variable_scope('decoder'):
                decoder_cell = self.init_RNN(self.num_units_lstm)
                outputs_decoder, state_decoder = tf.nn.dynamic_rnn(decoder_cell, x_decoder, dtype="float32",
                                                                   initial_state=state_encoder)
                outputs_decoder = tf.identity(outputs_decoder, name='outputs_decoder')
                prediction = outputs_decoder[:, :, -1]
                prediction = tf.reshape(prediction, (tf.shape(outputs_decoder)[0], 1, tf.shape(outputs_decoder)[1]))
            prediction = tf.identity(prediction, name='prediction')
            # loss_function
            if self.variant:
                # Kullback-Leibler divergence with 2 gaussian distribution when reparameterize is:
                # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1 -mu2)^2)/2*sigma2^2 - 1/2
                # => just for one dimension, we need to scale it up
                reconstruction_loss = tf.reduce_mean(tf.square(y_decoder - prediction))
                kl = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(2.0 * log_var) - 2.0 * log_var - 1.0)
                loss = reconstruction_loss + kl
            else:
                loss = tf.reduce_mean(tf.square(y_decoder - prediction))
            loss = tf.identity(loss, name='loss')
            graph = tf.get_default_graph()

            sess = tf.Session()
            # init = tf.global_variables_initializer()
            # sess.run(init)
            trainable_variables = tf.trainable_variables()
            trainable_tensor = []
            for i, v in enumerate(trainable_variables):
                v_tensor = graph.get_tensor_by_name(v.name)
                _initiate_value = np.random.uniform(-1, 1, v_tensor.shape)
                sess.run(tf.assign(v_tensor, _initiate_value))

                trainable_tensor.append(v_tensor)
            saver = tf.train.Saver()
            space.particles.append(ParticleAutoEncoder(graph, sess))
        print('>>> Create space complete, start training with pso <<<')
        best_autoencoder_graph, best_autoencoder_graph_fitness, cost_train_set, cost_valid_set = space.train()

        saver.save(best_autoencoder_graph.sess, self.model_saved_path)
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(best_autoencoder_graph_fitness) + '\n')
        self.draw_train_loss(cost_train_set, cost_valid_set)
        print("=== bess fitness autoencoder model: {} ===".format(best_autoencoder_graph_fitness))

    def __fit_with_bppso(self):
        tf.reset_default_graph()

        # define placeholder of graph
        x_encoder = tf.placeholder("float", [None, self.x_train_encoder.shape[1], self.x_train_encoder.shape[2]],
                                   name='x_encoder')
        x_decoder = tf.placeholder("float", [None, self.x_train_decoder.shape[1], self.x_train_decoder.shape[2]],
                                   name='x_decoder')
        y_decoder = tf.placeholder("float", [None, self.y_train_decoder.shape[1], self.y_train_decoder.shape[2]],
                                   name='y_decoder')

        with tf.variable_scope('encoder'):
            encoder_cell = self.init_RNN(self.num_units_lstm)
            outputs_encoder, state_encoder = tf.nn.dynamic_rnn(encoder_cell, x_encoder, dtype="float32")
            outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
            if self.variant:
                print('using variant for auto encoder!!!')
                mean = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_mean')
                log_var = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_log_var')
                batch = tf.shape(mean)[0]
                dim = tf.keras.backend.int_shape(mean)[1]
                # by default, random_normal has mean=0 and std=1.0
                epsilon = tf.random_normal(shape=(batch, dim))
                hidden_state_encoder = mean + tf.exp(0.5 * log_var) * epsilon
            else:
                hidden_state_encoder = state_encoder[-1].h
            hidden_state_encoder = tf.identity(hidden_state_encoder, name='hidden_state_encoder')
        with tf.variable_scope('decoder'):
            decoder_cell = self.init_RNN(self.num_units_lstm)
            outputs_decoder, state_decoder = tf.nn.dynamic_rnn(decoder_cell, x_decoder, dtype="float32",
                                                               initial_state=state_encoder)
            outputs_decoder = tf.identity(outputs_decoder, name='outputs_decoder')
            prediction = outputs_decoder[:, :, -1]
            prediction = tf.reshape(prediction, (tf.shape(outputs_decoder)[0], 1, tf.shape(outputs_decoder)[1]))
        prediction = tf.identity(prediction, name='prediction')
        # loss_function
        if self.variant:
            # Kullback-Leibler divergence with 2 gaussian distribution when reparameterize is:
            # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1 -mu2)^2)/2*sigma2^2 - 1/2
            # => just for one dimension, we need to scale it up
            reconstruction_loss = tf.reduce_mean(tf.square(y_decoder - prediction))
            kl = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(2.0 * log_var) - 2.0 * log_var - 1.0)
            loss = reconstruction_loss + kl
        else:
            loss = tf.reduce_mean(tf.square(y_decoder - prediction))
        loss = tf.identity(loss, name='loss')
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        cost_train_set = []
        cost_valid_set = []
        graph = tf.get_default_graph()
        # Initial space
        space = SpaceAutoEncoder(self.num_particle, self.x_train_encoder, self.x_train_decoder, self.y_train_decoder,
                                 self.x_valid_encoder, self.x_valid_decoder, self.y_valid_decoder, self.x_test_encoder,
                                 self.x_test_decoder, self.y_test_decoder, self.batch_size, self.epochs)
        space.particles = []
        # Training with back propagation and initial particle for space
        for particle_idx in range(self.num_particle):
            print('>>> Start training to initial weight for particle autoencoder {} <<<'.format(particle_idx + 1))
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()

            for epoch in range(self.epochs):
                start_time = time.time()
                total_batch = int(len(self.x_train_encoder) / self.batch_size)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder = self.x_train_encoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_xs_decoder = self.x_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys = self.y_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder,
                                                   y_decoder: batch_ys})

                    avg_cost += sess.run(loss, feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder,
                                                          y_decoder: batch_ys}) / total_batch

                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print('time: {}'.format(time.time() - start_time))
                cost_train_set.append(avg_cost)
                val_cost = sess.run(loss, feed_dict={x_encoder: self.x_valid_encoder,
                                                     x_decoder: self.x_valid_decoder,
                                                     y_decoder: self.y_valid_decoder})
                cost_valid_set.append(val_cost)
                if (epoch > self.patience):
                    if not self.early_stopping(cost_train_set, self.patience):
                        print('early stop training auto encoder model')
                        break

            space.particles.append(ParticleAutoEncoder(graph, sess))

        best_autoencoder_graph, best_autoencoder_graph_fitness = space.train()
        saver = tf.train.Saver()
        saver.save(best_autoencoder_graph.sess, self.model_saved_path)
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(best_autoencoder_graph_fitness) + '\n')
        print("=== bess fitness autoencoder model: {} ===".format(best_autoencoder_graph_fitness))

    def __fit_with_psobp(self):
        autoencoder_pso_file = Config.MODEL_SAVE_PATH.format('bnn/autoencoder', 'pso',
                                                             Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'],
                                                             Config.GOOGLE_TRACE_DATA_CONFIG['predict_data']) + self.file_name + '/model'
        evaluation_pso = Config.EVALUATION_PATH.format('bnn/autoencoder', 'pso',
                                                       Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'],
                                                       Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'])

        # if not os.path.exists(autoencoder_metadata):
        print('>>> [Note: ] This graph has not trained by pso algorithm!!! Train it first ...')
        print('>>> Start initiate space and particle <<<')
        # Initial space
        space = SpaceAutoEncoder(self.num_particle, self.x_train_encoder, self.x_train_decoder, 
                                 self.y_train_decoder, self.x_valid_encoder, self.x_valid_decoder, 
                                 self.y_valid_decoder, self.x_test_encoder, self.x_test_decoder, 
                                 self.y_test_decoder, self.batch_size, self.epochs)
        space.particles = []
        # Training with back propagation and initial particle for space
        for particle_idx in range(self.num_particle):
            tf.reset_default_graph()

            # define placeholder of graph
            x_encoder = tf.placeholder('float', [None, self.x_train_encoder.shape[1], self.x_train_encoder.shape[2]],
                                       name='x_encoder')
            x_decoder = tf.placeholder('float', [None, self.x_train_decoder.shape[1], self.x_train_decoder.shape[2]],
                                       name='x_decoder')
            y_decoder = tf.placeholder('float', [None, self.y_train_decoder.shape[1], self.y_train_decoder.shape[2]],
                                       name='y_decoder')

            with tf.variable_scope('encoder'):
                encoder_cell = self.init_RNN(self.num_units_lstm)
                outputs_encoder, state_encoder = tf.nn.dynamic_rnn(encoder_cell, x_encoder, dtype='float32')
                outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
                if self.variant:
                    mean = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_mean')
                    log_var = tf.layers.dense(state_encoder[-1].h, self.num_units_lstm[-1], name='z_log_var')
                    batch = tf.shape(mean)[0]
                    dim = tf.keras.backend.int_shape(mean)[1]
                    # by default, random_normal has mean=0 and std=1.0
                    epsilon = tf.random_normal(shape=(batch, dim))
                    hidden_state_encoder = mean + tf.exp(0.5 * log_var) * epsilon
                else:
                    hidden_state_encoder = state_encoder[-1].h
                hidden_state_encoder = tf.identity(hidden_state_encoder, name='hidden_state_encoder')
            with tf.variable_scope('decoder'):
                decoder_cell = self.init_RNN(self.num_units_lstm)
                outputs_decoder, state_decoder = tf.nn.dynamic_rnn(decoder_cell, x_decoder, dtype="float32",
                                                                   initial_state=state_encoder)
                outputs_decoder = tf.identity(outputs_decoder, name='outputs_decoder')
                prediction = outputs_decoder[:, :, -1]
                prediction = tf.reshape(prediction, (tf.shape(outputs_decoder)[0], 1, tf.shape(outputs_decoder)[1]))
            prediction = tf.identity(prediction, name='prediction')
            # loss_function
            if self.variant:
                # Kullback-Leibler divergence with 2 gaussian distribution when reparameterize is:
                # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1 -mu2)^2)/2*sigma2^2 - 1/2
                # => just for one dimension, we need to scale it up
                reconstruction_loss = tf.reduce_mean(tf.square(y_decoder - prediction))
                kl = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(2.0 * log_var) - 2.0 * log_var - 1.0)
                loss = reconstruction_loss + kl
            else:
                loss = tf.reduce_mean(tf.square(y_decoder - prediction))
            loss = tf.identity(loss, name='loss')
            # optimization
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            cost_train_set = []
            cost_valid_set = []
            graph = tf.get_default_graph()
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            space.particles.append(ParticleAutoEncoder(graph, sess))
        print('>>> Initiate space and particle complete !!! <<<')
        best_autoencoder_element, best_autoencoder_element_fitness, cost_train_set, cost_valid_set = space.train()
        saver.save(best_autoencoder_element.sess, autoencoder_pso_file)
        autoencoder_pso_metadata = autoencoder_pso_file + '.meta'
        autoencoder_pso_model_graph_dir = autoencoder_pso_metadata.rsplit(os.sep, 1)[0]
        # graph = best_autoencoder_element.graph
        # x_encoder = graph.get_tensor_by_name('x_encoder:0')
        # x_decoder = graph.get_tensor_by_name('x_decoder:0')
        # y_decoder = graph.get_tensor_by_name('y_decoder:0')
        # hidden_state_encoder = graph.get_tensor_by_name('encoder/hidden_state_encoder:0')
        # prediction = graph.get_tensor_by_name('prediction:0')
        # loss = graph.get_tensor_by_name('loss:0')
        # # optimization
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # sess = best_autoencoder_element.sess
        # # init value for optimization
        # var_list = [var for var in tf.global_variables()]
        # var_list_not_name = []
        # for var in var_list:
        #     try:
        #         sess.run(var)
        #     except:
        #         var_list_not_name.append(var)
        # var_list_init = tf.variables_initializer(var_list_not_name)
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            checkpoint = tf.train.import_meta_graph(autoencoder_pso_metadata)
            checkpoint.restore(sess, tf.train.latest_checkpoint(autoencoder_pso_model_graph_dir))

            x_encoder = graph.get_tensor_by_name('x_encoder:0')
            x_decoder = graph.get_tensor_by_name('x_decoder:0')
            y_decoder = graph.get_tensor_by_name('y_decoder:0')
            hidden_state_encoder = graph.get_tensor_by_name('encoder/hidden_state_encoder:0')
            prediction = graph.get_tensor_by_name('prediction:0')
            loss = graph.get_tensor_by_name('loss:0')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            var_list = [var for var in tf.global_variables()]
            var_list_not_name = []
            for var in var_list:
                try:
                    sess.run(var)
                except:
                    var_list_not_name.append(var)
            var_list_init = tf.variables_initializer(var_list_not_name)
            saver = tf.train.Saver()
            sess.run(var_list_init)

        # tf.get_default_graph()

        for epoch in range(self.epochs):
            start_time = time.time()
            total_batch = int(len(self.x_train_encoder) / self.batch_size)
            avg_cost = 0
            for i in range(total_batch):
                batch_xs_encoder = self.x_train_encoder[i * self.batch_size:(i + 1) * self.batch_size]
                batch_xs_decoder = self.x_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                batch_ys = self.y_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                sess.run(optimizer, feed_dict={x_encoder: batch_xs_encoder, 
                                               x_decoder: batch_xs_decoder,
                                               y_decoder: batch_ys})

                avg_cost += sess.run(loss, feed_dict={x_encoder: batch_xs_encoder, 
                                                      x_decoder: batch_xs_decoder,
                                                      y_decoder: batch_ys}) / total_batch
            training_history = 'Epoch autoencoder %04d: cost = %.9f with time: %.9f'\
                % (epoch + 1, avg_cost, time.time() - start_time)
            print(training_history)
            cost_train_set.append(avg_cost)
            val_cost = sess.run(
                loss, feed_dict={x_encoder: self.x_valid_encoder,
                                 x_decoder: self.x_valid_decoder,
                                 y_decoder: self.y_valid_decoder})
            cost_valid_set.append(val_cost)
            if epoch > self.patience:
                if self.early_stopping(cost_train_set, self.patience):
                    print('early stop training auto encoder model')
                    break
        test_loss = sess.run(loss, feed_dict={x_encoder: self.x_test_encoder, 
                                              x_decoder: self.x_test_decoder,
                                              y_decoder: self.y_test_decoder})
        saver.save(sess, self.model_saved_path)
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(test_loss) + '\n')
        self.draw_train_loss(cost_train_set, cost_valid_set)
        print("=== final avg cost: {} ===".format(test_loss))

    def fit(self):
        print(">>> Start training autoencoder model <<<")
        if self.optimizer_approach.lower() == 'bp':
            print(">>> Training autoencoder with back propagation <<<")
            self.__fit_with_bp()
        elif self.optimizer_approach.lower() == 'pso':
            print(">>> Training autoencoder with pso <<<")
            self.__fit_with_pso()
        elif self.optimizer_approach.lower() == 'bp_pso':
            print(">>> Training autoencoder with pso combined with back propagation <<<")
            self.__fit_with_bppso()
        elif self.optimizer_approach.lower() == 'pso_bp':
            print('>>> Training autoencoder with back propagation after use pso <<<')
            self.__fit_with_psobp()
        else:
            ">>> error: We don't support this optimzer approach! <<<"
