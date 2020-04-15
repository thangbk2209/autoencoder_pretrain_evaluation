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
from lib.bnn.autoencoder import AutoEncoder
from lib.evolution_algorithms.pso import *


matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class BnnPredictor:
    def __init__(self, data=None, scaler=None, train_size=None, valid_size=None, sliding_encoder=None,
                 sliding_inference=None, batch_size=None, num_units_lstm=None, num_units_inference=None,
                 dropout_rate=None, variation_dropout=False, activation=None, optimizer=None, variant=None,
                 optimizer_approach=None, learning_rate=None, epochs=None, patience=None, num_particle=None):
        self.data = data
        self.scaler = scaler
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units_lstm = num_units_lstm
        self.num_units_inference = num_units_inference
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
        self.create_name()
        if Config.DATA_EXPERIMENT == 'google_trace':
            self.x_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type']
            self.y_data_name = Config.GOOGLE_TRACE_DATA_CONFIG['predict_data']

        self.evaluation_path = Config.EVALUATION_PATH.format(
            'bnn/inference', self.optimizer_approach, self.x_data_name, self.y_data_name)
        self.model_saved_path = Config.MODEL_SAVE_PATH.format(
            'bnn/inference', self.optimizer_approach, self.x_data_name, self.y_data_name) + self.file_name + '/model'

        self.train_loss_path = Config.TRAIN_LOSS_PATH.format(
            'bnn/inference', self.optimizer_approach, self.x_data_name, self.y_data_name) + self.file_name + '.png'

        self.results_save_path = Config.RESULTS_SAVE_PATH.format(
            'bnn/inference', self.optimizer_approach, self.x_data_name, self.y_data_name) + self.file_name + '.csv'

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
        name_inference = create_name_network(self.num_units_inference)

        part1 = 'sli_encoder-{}_batch-{}_name_lstm-{}'.format(self.sliding_encoder, self.batch_size, name_lstm)
        part2 = 'activation-{}_optimizer_{}'.format(self.activation, self.optimizer)
        part4 = '_num_par-{}'.format(self.num_particle)
        part5 = '_name_inf-{}_sliding_inf-{}'.format(name_inference, self.sliding_inference)
        if self.variation_dropout:
            part3 = '_dropout_rate-{}'.format(self.dropout_rate)
        else:
            part3 = ''

        self.file_name = part1 + part2 + part3 + part4 + part5
        self.encoder_file_name = part1 + part2 + part3 + part4

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train_inference, self.y_train_inference, self.x_valid_inference, self.y_valid_inference, \
            self.x_test_inference, self.y_test_inference = data_preprocessor.init_data_inference(self.sliding_encoder,
                                                                                                 self.sliding_inference)

        print('>>> Init auto encoder model <<<')
        autoencoder = AutoEncoder(data=self.data, scaler=self.scaler, train_size=self.train_size,
                                  valid_size=self.valid_size, sliding_encoder=self.sliding_encoder,
                                  batch_size=self.batch_size, num_units_lstm=self.num_units_lstm,
                                  dropout_rate=self.dropout_rate, variation_dropout=self.variation_dropout,
                                  activation=self.activation, optimizer=self.optimizer, variant=self.variant,
                                  optimizer_approach=self.optimizer_approach, learning_rate=self.learning_rate,
                                  epochs=self.epochs, patience=self.patience, num_particle=self.num_particle)
        print('>>> Compute hidden state <<<')
        self.state_train, self.state_valid, self.state_test = autoencoder.compute_state_vector()

        self.x_train_inference = np.concatenate((self.x_train_inference, self.state_train), axis=1)
        self.x_valid_inference = np.concatenate((self.x_valid_inference, self.state_valid), axis=1)
        self.x_test_inference = np.concatenate((self.x_test_inference, self.state_test), axis=1)

    def draw_train_loss(self, cost_train_set, cost_valid_set):
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.train_loss_path)
        plt.close()

    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, num_units[i], activation=activation, name='layer' + str(i))
            drop_rate = self.dropout_rate
            prev_layer = tf.layers.dropout(prev_layer, rate=drop_rate)

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

    def __fit_with_bp(self):
        tf.reset_default_graph()

        # define placeholder of graph
        x_inference = tf.placeholder("float", [None, self.x_train_inference.shape[1]])
        y_inference = tf.placeholder("float", [None, self.y_train_inference.shape[1]])
        # state_encoder = tf.placeholder("float", [None, self.num_units_lstm[-1]])
        # input_inference = tf.concat([x_inference, state_encoder], 1)
        prediction = self.mlp(x_inference, self.num_units_inference, self.activation_func)

        loss = tf.reduce_mean(tf.square(y_inference - prediction))
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            # training bnn model
            print("start training bnn model")
            cost_train_set = []
            cost_valid_set = []
            for epoch in range(self.epochs):
                # Train with each example
                print('epoch BNN model: ', epoch + 1)
                total_batch = int(len(self.x_train_inference) / self.batch_size)
                print(total_batch)
                start_time = time.time()
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_inference = self.x_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys_inference = self.y_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={x_inference: batch_xs_inference,
                                                   y_inference: batch_ys_inference})

                    avg_cost += sess.run(loss, feed_dict={x_inference: batch_xs_inference,
                                                          y_inference: batch_ys_inference})
                # Display logs per epoch step
                training_history = 'Epoch inference %04d: cost = %.9f with time: %.2f'\
                    % (epoch + 1, avg_cost, time.time() - start_time)
                print(training_history)
                val_cost = sess.run(loss, feed_dict={x_inference: self.x_valid_inference,
                                                     y_inference: self.y_valid_inference})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)
                if epoch > self.patience:
                    if self.early_stopping(cost_train_set, self.patience):
                        print("early stopping BNN training process")
                        break
            saver.save(sess, self.model_saved_path)
            self.draw_train_loss(cost_train_set, cost_valid_set)
            print('training BNN with back propagation complete!!!')
            prediction = sess.run(prediction, feed_dict={x_inference: self.x_test_inference})
            prediction_inverse = self.scaler.inverse_transform(prediction)
            prediction_inverse = np.asarray(prediction_inverse)
            self.y_test_inversed = self.scaler.inverse_transform(self.y_test_inference)

            mae_err = MAE(prediction_inverse, self.y_test_inversed)
            rmse_err = np.sqrt(MSE(prediction_inverse, self.y_test_inversed))

            prediction_df = pd.DataFrame(np.array(prediction_inverse))
            prediction_df.to_csv(self.results_save_path, index=False, header=None)

            with open(self.evaluation_path, 'a+') as f:
                f.write(self.file_name + ',' + str(mae_err) + ',' + str(rmse_err) + '\n')

            print("=== error = {}, {} ===".format(mae_err, rmse_err))

            sess.close()

    def __fit_with_pso(self):
        print('>>>> Start training inference with pso algorithm <<<')

        space = Space(self.num_particle, self.x_train_inference, self.y_train_inference, self.x_valid_inference,
                      self.y_valid_inference, self.x_test_inference, self.y_test_inference, self.batch_size,
                      self.epochs)
        space.particles = []

        for _ in range(self.num_particle):
            tf.reset_default_graph()

            # define placeholder of graph
            x_inference = tf.placeholder("float", [None, self.x_train_inference.shape[1]], name='x')
            y_inference = tf.placeholder("float", [None, self.y_train_inference.shape[1]], name='y')
            prediction = self.mlp(x_inference, self.num_units_inference, self.activation_func)
            prediction = tf.identity(prediction, name='prediction')
            loss = tf.reduce_mean(tf.square(y_inference - prediction))
            loss = tf.identity(loss, name='loss')
            graph = tf.get_default_graph()
            sess = tf.Session()
            trainable_variables = tf.trainable_variables()
            trainable_tensor = []
            # self.update_tensor_opts = []
            for i, v in enumerate(trainable_variables):
                v_tensor = graph.get_tensor_by_name(v.name)
                _initiate_value = np.random.uniform(-1, 1, v_tensor.shape)
                sess.run(tf.assign(v_tensor, _initiate_value))

                trainable_tensor.append(v_tensor)
            # init = tf.global_variables_initializer()
            # sess.run(init)
            saver = tf.train.Saver()
            space.particles.append(Particle(graph, sess))
        best_particle, prediction, cost_train_set, cost_valid_set = space.train()
        inversed_prediction = self.scaler.inverse_transform(prediction)
        inversed_prediction = np.asarray(inversed_prediction)
        self.y_test_inversed = self.scaler.inverse_transform(self.y_test_inference)

        prediction_df = pd.DataFrame(np.array(inversed_prediction))
        prediction_df.to_csv(self.results_save_path, index=False, header=None)

        mae_err = MAE(inversed_prediction, self.y_test_inversed)
        rmse_err = np.sqrt(MSE(inversed_prediction, self.y_test_inversed))
        print("=== error inference model: {}, {} ===".format(mae_err, rmse_err))
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(mae_err) + ',' + str(rmse_err) + '\n')
        saver.save(best_particle.sess, self.model_saved_path)
        self.draw_train_loss(cost_train_set, cost_valid_set)

    def __fit_with_bppso(self):
        tf.reset_default_graph()

        # define placeholder of graph
        # x_inference = tf.placeholder("float", [None, self.x_train_inference.shape[1]])
        # y_inference = tf.placeholder("float", [None, self.y_train_inference.shape[1]])
        x_inference = tf.placeholder("float", [None, self.x_train_inference.shape[1]], name='x')
        y_inference = tf.placeholder("float", [None, self.y_train_inference.shape[1]], name='y')

        prediction = self.mlp(x_inference, self.num_units_inference, self.activation_func)
        prediction = tf.identity(prediction, name='prediction')
        loss = tf.reduce_mean(tf.square(y_inference - prediction))
        loss = tf.identity(loss, name='loss')
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        graph = tf.get_default_graph()
        space = Space(self.num_particle, self.x_train_inference, self.y_train_inference, self.x_valid_inference,
                      self.y_valid_inference, self.x_test_inference, self.y_test_inference, self.batch_size,
                      self.epochs)
        space.particles = []
        for particle_idx in range(self.num_particle):
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            # training bnn model
            print("start training bnn model")
            cost_train_set = []
            cost_valid_set = []
            for epoch in range(self.epochs):
                # Train with each example
                total_batch = int(len(self.x_train_inference) / self.batch_size)
                start_time = time.time()
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_inference = self.x_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_ys_inference = self.y_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={x_inference: batch_xs_inference,
                                                   y_inference: batch_ys_inference})

                    avg_cost += sess.run(loss, feed_dict={x_inference: batch_xs_inference,
                                                          y_inference: batch_ys_inference})
                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print('time: {}'.format(time.time() - start_time))
                val_cost = sess.run(loss, feed_dict={x_inference: self.x_valid_inference,
                                                     y_inference: self.y_valid_inference})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)
                if (epoch > self.patience):
                    if not self.early_stopping(cost_train_set, self.patience):
                        print("early stopping BNN training process")
                        break

            space.particles.append(Particle(graph, sess))
        best_particle, prediction = space.train()
        inversed_prediction = self.scaler.inverse_transform(prediction)
        inversed_prediction = np.asarray(inversed_prediction)
        self.y_test_inversed = self.scaler.inverse_transform(self.y_test_inference)
        saver.save(best_particle.sess, self.model_saved_path)
        prediction_df = pd.DataFrame(np.array(inversed_prediction))
        prediction_df.to_csv(self.results_save_path, index=False, header=None)

        mae_err = MAE(inversed_prediction, self.y_test_inversed)
        rmse_err = np.sqrt(MSE(inversed_prediction, self.y_test_inversed))
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(mae_err) + ',' + str(rmse_err) + '\n')
        print("=== error inference model: {}, {} ===".format(mae_err, rmse_err))

    def __fit_with_psobp(self):
        print('>>>> Start training inference with pso bp algorithm <<<')
        inf_pso_file = Config.MODEL_SAVE_PATH.format(
            'bnn/inference', 'pso', self.x_data_name, self.y_data_name) + self.file_name + '/model'
        # Training bnn with pso
        space = Space(self.num_particle, self.x_train_inference, self.y_train_inference, self.x_valid_inference,
                      self.y_valid_inference, self.x_test_inference, self.y_test_inference, self.batch_size,
                      self.epochs)
        space.particles = []
        for particle_idx in range(self.num_particle):
            tf.reset_default_graph()

            # define placeholder of graph
            x_inference = tf.placeholder("float", [None, self.x_train_inference.shape[1]], name='x')
            y_inference = tf.placeholder("float", [None, self.y_train_inference.shape[1]], name='y')

            prediction = self.mlp(x_inference, self.num_units_inference, self.activation_func)
            prediction = tf.identity(prediction, name='prediction')
            loss = tf.reduce_mean(tf.square(y_inference - prediction))
            loss = tf.identity(loss, name='loss')
            # optimization
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            graph = tf.get_default_graph()
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            space.particles.append(Particle(graph, sess))
        best_particle, prediction, cost_train_set, cost_valid_set = space.train()
        saver.save(best_particle.sess, inf_pso_file)
        inf_pso_metadata = inf_pso_file + '.meta'
        inf_pso_model_graph_dir = inf_pso_metadata.rsplit(os.sep, 1)[0]
        sess = best_particle.sess
        print('>>> Training inference model by pso complete!!! Start training with bp <<<')
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            checkpoint = tf.train.import_meta_graph(inf_pso_metadata)
            checkpoint.restore(sess, tf.train.latest_checkpoint(inf_pso_model_graph_dir))

            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')
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
        # More optimization with back propagation
        # cost_train_set = []
        # cost_valid_set = []
        for epoch in range(self.epochs):
            # Train with each example
            total_batch = int(len(self.x_train_inference) / self.batch_size)
            start_time = time.time()
            avg_cost = 0
            for i in range(total_batch):
                batch_xs_inference = self.x_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                batch_ys_inference = self.y_train_inference[i * self.batch_size:(i + 1) * self.batch_size]
                sess.run(optimizer, feed_dict={x: batch_xs_inference,
                                               y: batch_ys_inference})

                avg_cost += sess.run(loss, feed_dict={x: batch_xs_inference,
                                                      y: batch_ys_inference})
            # Display logs per epoch step
            training_history = 'Epoch inference %04d: cost = %.9f with time: %.2f'\
                % (epoch + 1, avg_cost, time.time() - start_time)
            print(training_history)
            val_cost = sess.run(loss, feed_dict={x: self.x_valid_inference,
                                                 y: self.y_valid_inference})

            cost_train_set.append(avg_cost)
            cost_valid_set.append(val_cost)
            if epoch > self.patience:
                if self.early_stopping(cost_train_set, self.patience):
                    print("early stopping BNN training process")
                    break
        # _x_inference = graph.get_tensor_by_name('x:0')
        # _prediction = graph.get_tensor_by_name('prediction:0')
        y_prediction = sess.run(prediction, feed_dict={x: self.x_test_inference})

        inversed_prediction = self.scaler.inverse_transform(y_prediction)
        inversed_prediction = np.asarray(inversed_prediction)
        self.y_test_inversed = self.scaler.inverse_transform(self.y_test_inference)
        saver.save(sess, self.model_saved_path)
        prediction_df = pd.DataFrame(np.array(inversed_prediction))
        prediction_df.to_csv(self.results_save_path, index=False, header=None)

        mae_err = MAE(inversed_prediction, self.y_test_inversed)
        rmse_err = np.sqrt(MSE(inversed_prediction, self.y_test_inversed))
        with open(self.evaluation_path, 'a+') as f:
            f.write(self.file_name + ',' + str(mae_err) + ',' + str(rmse_err) + '\n')
        self.draw_train_loss(cost_train_set, cost_valid_set)
        print("=== error inference model: {}, {} ===".format(mae_err, rmse_err))

    def fit(self):
        training_time_evaluation_file = CORE_DATA_DIR + '/bnn/training_time_evaluation.csv'
        start_all_time = time.time()

        print(">>> Start training bnn model <<<")
        self.preprocessing_data()
        if self.optimizer_approach.lower() == 'bp':
            print(">>> Training bnn with back propagation <<<")
            self.__fit_with_bp()
        elif self.optimizer_approach.lower() == 'pso':
            print(">>> Training bnn with pso <<<")
            self.__fit_with_pso()
        elif self.optimizer_approach.lower() == 'bp_pso':
            print(">>> Training bnn with pso combined with back propagation <<<")
            self.__fit_with_bppso()
        elif self.optimizer_approach.lower() == 'pso_bp':
            print(">>> Training bnn with pso combined with back propagation <<<")
            self.__fit_with_psobp()
        else:
            ">>> error: We don't support this optimzer approach! <<<"
        with open(training_time_evaluation_file, 'a+') as f:
            f.write(self.model_saved_path + ',' + str(time.time() - start_all_time) + '\n')