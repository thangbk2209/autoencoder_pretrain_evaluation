import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as r2

from lib.preprocess_data.preprocessing_data import DataPreprocessor
from config import *

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class LstmPredictor:
    def __init__(self, data=None, scaler=None, sliding=None, batch_size=None, num_units=None, dropout_rate=None,
                 variation_dropout=False, activation=None, optimizer=None, optimizer_approach=None, learning_rate=None,
                 epochs=None, early_stopping=None, patience=None, model_save_path=None, results_save_path=None,
                 train_size=None, valid_size=None, train_loss_path=None, evaluation_path=None, time_list=None):
        self.data = data
        self.scaler = scaler
        self.sliding = sliding
        self.batch_size = batch_size
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.variation_dropout = variation_dropout
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_approach = optimizer_approach
        self.lr = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience

        self.train_size = train_size
        self.valid_size = valid_size
        self.time_list = time_list
        self.create_name()
        self.train_loss_path = train_loss_path
        self.evaluation_path = evaluation_path
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path

    def create_name(self):
        name_LSTM = ""
        for i in range(len(self.num_units)):
            if (i == len(self.num_units) - 1):
                name_LSTM += str(self.num_units[i])
            else:
                name_LSTM += str(self.num_units[i]) + '_'

        part1 = 'sli-{}_batch-{}'.format(self.sliding, self.batch_size)
        part2 = '_numunits-{}_act-{}_opt_{}'.format(name_LSTM, self.activation, self.optimizer)
        if self.optimizer_approach == 'bp':
            part4 = ''
        elif self.optimizer_approach == 'pso':
            part4 = '_num_par-{}_w_old_velocity-{}_w_local_best_position-{}_w_global_best_position-{}'\
                .format(self.num_particle, self.w_old_velocity, self.w_local_best_position, self.w_global_best_position)
        elif self.optimizer_approach == 'whale':
            part4 = '_num_par-{}'.format(self.num_particle)

        if self.variation_dropout:
            part3 = '_drop_rate-{}'.format(self.dropout_rate)
        else:
            part3 = ''
        part5 = f'_time_{self.time_list}'

        self.file_name = part1 + part2 + part3 + part4 + part5

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train, self.y_train, self.x_valid, self.y_valid, \
            self.x_test, self.y_test = data_preprocessor.init_data_lstm(self.sliding)

    def init_RNN(self, num_units, activation):
        num_layers = len(num_units)
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                cell = tf.nn.rnn_cell.LSTMCell(num_units[i], activation=activation)
                if self.variation_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                                                         state_keep_prob=self.dropout_rate, variational_recurrent=True,
                                                         input_size=self.x_train.shape[2], dtype=tf.float32)
                hidden_layers.append(cell)
            else:
                cell = tf.nn.rnn_cell.LSTMCell(num_units[i], activation=activation)
                if self.variation_dropout:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                                                         state_keep_prob=self.dropout_rate, variational_recurrent=True,
                                                         input_size=num_units[i - 1], dtype=tf.float32)
                hidden_layers.append(cell)
        rnn_cells = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple=True)
        return rnn_cells

    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, num_units[i], activation=activation, name='layer' + str(i))

            # prev_layer = tf.layers.dropout(prev_layer)

        prediction = tf.layers.dense(inputs=prev_layer, units=1, activation=activation, name='output_layer')
        return prediction

    def early_stopping_desition(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array) - patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if check == patience:
            return False
        else:
            return True

    def draw_train_loss(self, cost_train_set, cost_valid_set):
        print(self.train_loss_path + self.file_name + '.png')
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        
        plt.savefig(self.train_loss_path + self.file_name + '.png')
        plt.close()

    def compute_error(self, y_pred, y_true):
        MAE_err = MAE(y_pred, y_true)
        RMSE_err = np.sqrt(MSE(y_pred, y_true))
        r2_err = r2(y_pred, y_true)
        return MAE_err, RMSE_err, r2_err

    def __fit_with_bp(self):
        if self.activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif self.activation == 'relu':
            activation = tf.nn.relu
        elif self.activation == 'tanh':
            activation = tf.nn.tanh
        elif self.activation == 'elu':
            activation = tf.nn.elu
        else:
            print(">>> Can not apply your activation <<<")

        if self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        else:
            print(">>> Can not apply your optimizer <<<")

        tf.reset_default_graph()
        x = tf.placeholder("float", [None, self.sliding, self.x_train.shape[2]])
        y = tf.placeholder("float", [None, self.y_train.shape[1]])
        with tf.variable_scope('LSTM'):
            lstm_layer = self.init_RNN(self.num_units, activation)
            outputs, new_state = tf.nn.dynamic_rnn(lstm_layer, x, dtype="float32")
            outputs = tf.identity(outputs, name='outputs')
        hidden_state = new_state[-1].h
        prediction = self.mlp(hidden_state, [4], activation)
        # prediction = tf.layers.dense(outputs[:, :, -1], self.y_train.shape[1], activation=activation, use_bias=True)
        loss = tf.reduce_mean(tf.square(y - prediction))
        optimize = optimizer.minimize(loss)

        cost_train_set = []
        cost_valid_set = []
        epoch_set = []

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            print(">>> Start training with lstm <<<")
            for epoch in range(self.epochs):
                start_time = time.time()
                total_batch = int(len(self.x_train) / self.batch_size)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimize, feed_dict={x: batch_xs, y: batch_ys})
                    avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                val_cost = sess.run(loss, feed_dict={x: self.x_valid, y: self.y_valid})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)

                if self.early_stopping:
                    if epoch > self.patience:
                        if not self.early_stopping_desition(cost_train_set, self.patience):
                            print(">>>Early stopping training<<<")
                            break
                print('Epoch {}: cost = {} with time = {}'.format(epoch + 1, avg_cost, time.time() - start_time))
            print('>>> Training model done <<<')
            # draw training process loss
            self.draw_train_loss(cost_train_set, cost_valid_set)
            # compute prediction and error to evaluate model
            prediction = sess.run(prediction, feed_dict={x: self.x_test})
            inversed_prediction = self.scaler.inverse_transform(prediction)
            inversed_prediction = np.asarray(inversed_prediction)
            self.y_test_inversed = self.scaler.inverse_transform(self.y_test)
            MAE_err, RMSE_err, r2_err = self.compute_error(inversed_prediction, self.y_test_inversed)
            print("=== error: MAE = {}, RMSE = {},R2 = {} ===".format(MAE_err, RMSE_err, r2_err))
            # save prediction
            prediction_file = self.results_save_path + self.file_name + '.csv'
            predictionDf = pd.DataFrame(np.array(inversed_prediction))
            predictionDf.to_csv(prediction_file, index=False, header=None)
            # save model
            saver.save(sess, self.model_save_path + self.file_name + '/model')
            summary = open(self.evaluation_path, 'a+')
            summary.write('{}, {}, {}, {}\n'.format(self.file_name, MAE_err, RMSE_err, r2_err))

    def fit(self):
        training_time_evaluation_file = CORE_DATA_DIR + '/lstm/training_time_evaluation.csv'
        start_all_time = time.time()
        print(">>> Start training with LSTM <<<")
        self.preprocessing_data()
        if self.optimizer_approach.lower() == 'bp':
            print(">>> Training LSTM with back propagation <<<")
            self.__fit_with_bp()
        else:
            ">>> error: We don't support this optimzer approach! <<<"
        with open(training_time_evaluation_file, 'a+') as f:
            f.write(self.model_save_path + self.file_name + ',' + str(time.time() - start_all_time) + '\n')
