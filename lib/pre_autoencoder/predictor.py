import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from tensorflow.contrib import rnn
import time

from config import *
from lib.preprocess_data.preprocessing_data import DataPreprocessor
from lib.pre_autoencoder.pre_autoencoder import PreAutoEncoder

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class PreAutoEncoderPredictor:
    def __init__(self, data=None, scaler=None, train_size=None, valid_size=None, sliding_encoder=None, batch_size=None, 
                 num_units_lstm=None, num_units_inference=None, dropout_rate=None, variation_dropout=False, activation=None, optimizer=None, 
                 variant=None, optimizer_approach=None, learning_rate=None, epochs=None, patience=None, time_list=None):
        self.data = data
        self.scaler = scaler
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        # self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units_lstm = num_units_lstm
        self.num_units_inference = num_units_inference

        self.dropout_rate = dropout_rate
        self.variation_dropout = variation_dropout
        self.optimizer_approach = optimizer_approach
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.time_list = time_list

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
        self.create_name()
        if Config.DATA_EXPERIMENT == 'google_trace':
            self.x_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type']
            self.y_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['predict_data']

        self.evaluation_path = Config.EVALUATION_PATH.format(
            Config.MODEL_EXPERIMENT, self.optimizer_approach, self.x_data_name, self.y_data_name)
        
        self.model_saved_path = Config.MODEL_SAVE_PATH.format(
            Config.MODEL_EXPERIMENT, self.optimizer_approach, self.x_data_name, self.y_data_name) 
        if not os.path.exists(self.model_saved_path):
            os.mkdir(self.model_saved_path)
        
        self.model_saved_path += self.file_name
        if not os.path.exists(self.model_saved_path):
            os.mkdir(self.model_saved_path)

        # self.model_saved_path += '/model'

        self.train_loss_path = Config.TRAIN_LOSS_PATH.format(
            Config.MODEL_EXPERIMENT, self.optimizer_approach, self.x_data_name, self.y_data_name) + self.file_name

        self.results_save_path = Config.RESULTS_SAVE_PATH.format(
            Config.MODEL_EXPERIMENT, self.optimizer_approach, self.x_data_name, self.y_data_name) + self.file_name + '.csv'

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

        part1 = 'sli_encoder-{}_batch-{}_name_lstm-{}'.format(self.sliding_encoder, self.batch_size, name_lstm)
        part2 = 'activation-{}_optimizer_{}'.format(self.activation, self.optimizer)
        if self.variation_dropout:
            part3 = '_dropout_rate-{}'.format(self.dropout_rate)
        else:
            part3 = ''

        part4 = f'_time_{self.time_list}'

        self.file_name = part1 + part2 + part3 + part4
        self.encoder_file_name = part1 + part2 + part3 + part4

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train_encoder, self.x_train_decoder, self.y_train_decoder, self.x_valid_encoder, self.x_valid_decoder,\
            self.y_valid_decoder, self.x_test_encoder, self.x_test_decoder, self.y_test_decoder\
            = data_preprocessor.init_data_autoencoder(self.sliding_encoder)
        self.y_train, self.y_valid, self.y_test = data_preprocessor.init_data_inference(self.sliding_encoder)

    def draw_train_loss(self, cost_train_set, cost_valid_set, model_name):
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.train_loss_path + model_name + '.png')
        plt.close()

    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, num_units[i], activation=activation, name='layer' + str(i))
            # drop_rate = self.dropout_rate
            # prev_layer = tf.layers.dropout(prev_layer, rate=drop_rate)

        prediction = tf.layers.dense(inputs=prev_layer, units=1, activation=activation)
        return prediction

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

    def __fit_with_bp(self):
        self.n_input_encoder = self.x_train_encoder.shape[1]
        self.n_input_decoder = self.x_train_decoder.shape[1]
        self.n_output_inference = self.y_train.shape[1]

        tf.reset_default_graph()
        # x1 = tf.placeholder("float", [None, self.sliding_encoder * len(self.original_data) / self.input_dim, self.input_dim])
        # x2 = tf.placeholder("float",shape = (None, self.sliding_decoder*len(self.original_data)/self.input_dim, self.input_dim))
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
            prediction_autoencoder = outputs_decoder[:, :, -1]
            prediction_autoencoder = tf.reshape(
                prediction_autoencoder, (tf.shape(outputs_decoder)[0], 1, tf.shape(outputs_decoder)[1]))
        prediction_autoencoder = tf.identity(prediction_autoencoder, name='prediction_autoencoder')
        # loss_function
        if self.variant:
            # Kullback-Leibler divergence with 2 gaussian distribution when reparameterize is:
            # kl = log(sigma2/sigma1) + (sigma1^2 + (mu1 -mu2)^2)/2*sigma2^2 - 1/2
            # => just for one dimension, we need to scale it up
            reconstruction_loss = tf.reduce_mean(tf.square(y_decoder - prediction_autoencoder))
            kl = 0.5 * tf.reduce_mean(tf.square(mean) + tf.exp(2.0 * log_var) - 2.0 * log_var - 1.0)
            loss_autoencoder = reconstruction_loss + kl
        else:
            loss_autoencoder = tf.reduce_mean(tf.square(y_decoder - prediction_autoencoder))
        loss_autoencoder = tf.identity(loss_autoencoder, name='loss')
        # optimization
        optimizer_autoencoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_autoencoder)
        
        y_inf = tf.placeholder("float", [None, self.y_train.shape[1]])
        
        prediction_inf = self.mlp(hidden_state_encoder, self.num_units_inference, self.activation_func)
        loss_inf = tf.reduce_mean(tf.square(y_inf - prediction_inf))
        # optimization
        optimizer_inf = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_inf)
        
        cost_train_set = []
        cost_valid_set = []
        
        init = tf.global_variables_initializer()
        # try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            print("start training autoencoder model")
            for epoch in range(self.epochs):
                # print(f'epoch {epoch}:')
                # Train with each example
                start_time = time.time()
                total_batch = int(len(self.x_train_encoder) / self.batch_size)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder = self.x_train_encoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_xs_decoder = self.x_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys = self.y_train_decoder[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer_autoencoder, 
                            feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder, y_decoder: batch_ys})

                    avg_cost += sess.run(loss_autoencoder, 
                                        feed_dict={x_encoder: batch_xs_encoder, x_decoder: batch_xs_decoder,
                                                    y_decoder: batch_ys}) / total_batch
                # Display logs per epoch step
                training_history = 'Epoch autoencoder %04d: cost = %.9f with time: %.9f'\
                    % (epoch + 1, avg_cost, time.time() - start_time)
                print(training_history)
                cost_train_set.append(avg_cost)
                val_cost = sess.run(loss_autoencoder,
                                    feed_dict={x_encoder: self.x_valid_encoder, x_decoder: self.x_valid_decoder,
                                            y_decoder: self.y_valid_decoder})
                cost_valid_set.append(val_cost)
                if epoch > self.patience:
                    if self.early_stopping(cost_train_set, self.patience):
                        print('early stop training auto encoder model')
                        break

            if not os.path.exists(self.model_saved_path + '/autoencoder/model/'):
                os.mkdir(self.model_saved_path + '/autoencoder')
                os.mkdir(self.model_saved_path + '/autoencoder/model/')
            saver.save(sess, self.model_saved_path + '/autoencoder/model/model')
            self.draw_train_loss(cost_train_set, cost_valid_set, '_autoencoder')

            # training bnn model
            print("start training bnn model")
            cost_train_set = []
            cost_valid_set = []
            for epoch in range(self.epochs):
                # Train with each example
                total_batch = int(len(self.x_train_encoder) / self.batch_size)

                start_time = time.time()
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder = self.x_train_encoder[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys_inf = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer_inf,
                            feed_dict={x_encoder: batch_xs_encoder, y_inf: batch_ys_inf})

                    avg_cost += sess.run(loss_inf,
                                        feed_dict={x_encoder: batch_xs_encoder, y_inf: batch_ys_inf})
                # Display logs per epoch step
                training_history = 'Epoch inference %04d: cost = %.9f with time: %.2f'\
                    % (epoch + 1, avg_cost, time.time() - start_time)
                print(training_history)
                val_cost = sess.run(loss_inf, feed_dict={x_encoder: self.x_valid_encoder, y_inf: self.y_valid})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)
                if epoch > self.patience:
                    if self.early_stopping(cost_train_set, self.patience):
                        print("early stopping pretrain_predictor training process")
                        break
            
            if not os.path.exists(self.model_saved_path + '/predictor/model/'):
                os.mkdir(self.model_saved_path + '/predictor')
                os.mkdir(self.model_saved_path + '/predictor/model/')
            saver.save(sess, self.model_saved_path + '/predictor/model/model')
            self.draw_train_loss(cost_train_set, cost_valid_set, '_predictor')
            print('training BNN with back propagation complete!!!')
            prediction_inf = sess.run(prediction_inf, feed_dict={x_encoder: self.x_test_encoder})
            sess.close()
            prediction_inverse = self.scaler.inverse_transform(prediction_inf)
            prediction_inverse = np.asarray(prediction_inverse)
            self.y_test_inversed = self.scaler.inverse_transform(self.y_test)

            mae_err = MAE(prediction_inverse, self.y_test_inversed)
            rmse_err = np.sqrt(MSE(prediction_inverse, self.y_test_inversed))

            prediction_df = pd.DataFrame(np.array(prediction_inverse))
            prediction_df.to_csv(self.results_save_path, index=False, header=None)

            with open(self.evaluation_path, 'a+') as f:
                f.write(self.file_name + ',' + str(mae_err) + ',' + str(rmse_err) + '\n')

            print("=== error = {}, {} ===".format(mae_err, rmse_err))

            return rmse_err
        # except:
        #     return 1

    def fit(self):
        # training_time_evaluation_file = CORE_DATA_DIR + '/bnn/training_time_evaluation.csv'
        # start_all_time = time.time()

        print(">>> Start training bnn model <<<")
        self.preprocessing_data()
        if self.optimizer_approach.lower() == 'bp':
            print(">>> Training bnn with back propagation <<<")
            self.__fit_with_bp()
        else:
            ">>> error: We don't support this optimzer approach! <<<"
        # with open(training_time_evaluation_file, 'a+') as f:
        #     f.write(self.model_saved_path + ',' + str(time.time() - start_all_time) + '\n')