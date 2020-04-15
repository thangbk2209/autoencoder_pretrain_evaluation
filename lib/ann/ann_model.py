import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from lib.preprocess_data.preprocessing_data import DataPreprocessor
from config import *

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class AnnPredictor:
    def __init__(self, data=None, scaler=None, sliding=None, batch_size=None, num_units=None, activation=None,
                 optimizer=None, optimizer_approach=None, learning_rate=.1, epochs=None, early_stopping=None,
                 patience=None, model_save_path=None, results_save_path=None, train_size=None, valid_size=None,
                 num_particle=50, w_old_velocity=None, w_local_best_position=None, w_global_best_position=None,
                 train_loss_path=None, evaluation_path=None):
        self.data = data
        self.scaler = scaler
        self.sliding = sliding
        self.batch_size = batch_size
        self.num_units = num_units
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_approach = optimizer_approach
        self.lr = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.train_size = train_size
        self.valid_size = valid_size
        self.num_particle = num_particle
        self.w_old_velocity = w_old_velocity
        self.w_local_best_position = w_local_best_position
        self.w_global_best_position = w_global_best_position
        self.create_name()
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path
        self.train_loss_path = train_loss_path
        self.evaluation_path = evaluation_path

    def create_name(self):
        name = ""
        for i in range(len(self.num_units)):
            if (i == len(self.num_units) - 1):
                name += str(self.num_units[i])
            else:
                name += str(self.num_units[i]) + '_'

            part1 = 'sliding-{}_batch-{}'.format(self.sliding, self.batch_size)
            part2 = '_numunits-{}_activation-{}_optimizer-{}'.format(name, self.activation, self.optimizer)
            self.file_name = part1 + part2

    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, num_units[i], activation=activation, name='layer' + str(i))
            prev_layer = tf.layers.dropout(prev_layer)

        prediction = tf.layers.dense(inputs=prev_layer, units=1, activation=activation)
        return prediction

    def early_stopping_desition(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array) - patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if(check == patience):
            return False
        else:
            return True

    def draw_train_loss(self, cost_train_set, cost_valid_set):
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.train_loss_path + self.file_name + '.png')
        plt.close()

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train, self.y_train, self.x_valid, self.y_valid, \
            self.x_test, self.y_test = data_preprocessor.init_data_ann(self.sliding)

    def fit_with_bp(self):

        if self.activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif self.activation == 'relu':
            activation = tf.nn.relu
        elif self.activation == 'tanh':
            activation = tf.nn.tanh
        elif self.activation == 'elu':
            activation_func = tf.nn.elu
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
        x = tf.placeholder("float", [None, self.x_train.shape[1]])
        y = tf.placeholder("float", [None, self.y_train.shape[1]])
        prediction = self.mlp(x, self.num_units, activation)
        loss = tf.reduce_mean(tf.square(y - prediction))
        optimize = optimizer.minimize(loss)
        cost_train_set = []
        cost_valid_set = []
        epoch_set = []

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            print(">>> Start training with ann <<<")
            for epoch in range(self.epochs):
                start_time = time.time()
                print('>>> epoch : ', epoch + 1)
                total_batch = int(len(self.x_train) / self.batch_size)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs = self.x_train[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimize, feed_dict={x: batch_xs, y: batch_ys})
                    avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
                print("Cost=", "{:.9f}".format(avg_cost))
                val_cost = sess.run(loss, feed_dict={x: self.x_valid, y: self.y_valid})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)

                if self.early_stopping:
                    if epoch > self.patience:
                        if not self.early_stopping_desition(cost_train_set, self.patience):
                            print(">>>Early stopping training<<<")
                            break
                print('Time for epoch {}: {}'.format(epoch + 1, time.time() - start_time))
                print("Epoch {} finished <<<".format(epoch + 1))
            print('>>> Training model complete <<<')
            # draw training process loss
            self.draw_train_loss(cost_train_set, cost_valid_set)
            # compute prediction and error to evaluate model
            prediction = sess.run(prediction, feed_dict={x: self.x_test})
            inversed_prediction = self.scaler.inverse_transform(prediction)
            inversed_prediction = np.asarray(inversed_prediction)
            self.y_test_inversed = self.scaler.inverse_transform(self.y_test)
            MAE_err = MAE(inversed_prediction, self.y_test_inversed)
            RMSE_err = np.sqrt(MSE(inversed_prediction, self.y_test_inversed))
            # save prediction
            prediction_file = self.results_save_path + self.file_name + '.csv'
            prediction_df = pd.DataFrame(np.array(inversed_prediction))
            prediction_df.to_csv(prediction_file, index=False, header=None)
            # save model
            saver.save(sess, self.model_save_path + self.file_name + '/model')
            summary = open(self.evaluation_path, 'a+')
            summary.write('{}, {}, {}\n'.format(self.file_name, MAE_err, RMSE_err))

    def fit(self):
        self.preprocessing_data()
        if self.optimizer_approach.lower() == 'bp':
            self.fit_with_bp()
        else:
            ">>> error: We don't support this optimzer approach! <<<"
