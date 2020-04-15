import numpy as np
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *


class DataPreprocessor:
    def __init__(self, data=None, train_size=None, valid_size=None):
        self.data = data
        self.train_size = train_size
        self.valid_size = valid_size

    def create_timeseries(self, X):
        if len(X) > 1:
            data = np.concatenate((X[0], X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2, len(X), 1):
                    data = np.column_stack((data, X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                data.append(X[0][i])
            data = np.array(data)
        return data

    def create_x(self, timeseries, sliding):
        dataX = []
        for i in range(len(timeseries) - sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i + j])
            dataX.append(datai)
        return dataX

    def init_data_lstm(self, sliding):
        print('>>> start init data for training LSTM model <<<')

        if Config.DATA_EXPERIMENT == 'google_trace':
            if Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'cpu_mem':
                x_data = [self.data['cpu'], self.data['mem']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_cpu':
                x_data = [self.data['cpu']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_mem':
                x_data = [self.data['mem']]

            if Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'cpu':
                y_data = self.data['cpu']
            elif Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'mem':
                y_data = self.data['mem']
        elif Config.DATA_EXPERIMENT == 'grid':
            x_data = [self.data]
            y_data = self.data
        elif Config.DATA_EXPERIMENT == 'traffic':
            x_data = [self.data]
            y_data = self.data
        else:
            print("We do not support your data for preprocessing!")

        x_timeseries = self.create_timeseries(x_data)
        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)
        valid_point = int((self.train_size + self.valid_size) * num_points)

        x_sampple = self.create_x(x_timeseries, sliding)

        x_train = x_sampple[0:train_point - sliding]
        x_train = np.array(x_train)

        x_valid = x_sampple[train_point - sliding: valid_point - sliding]
        x_valid = np.array(x_valid)

        x_test = x_sampple[valid_point - sliding:]
        x_test = np.array(x_test)

        y_train = y_data[sliding: train_point]
        y_train = np.array(y_train)

        y_valid = y_data[train_point: valid_point]
        y_valid = np.array(y_valid)

        y_test = y_data[valid_point:]
        y_test = np.array(y_test)

        print('>>> Init data for training LSTM model done <<<')
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def init_data_ann(self, sliding):
        print('>>> start init data for training ANN model <<<')

        if Config.DATA_EXPERIMENT == 'google_trace':
            if Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'cpu_mem':
                x_data = [self.data['cpu'], self.data['mem']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_cpu':
                x_data = [self.data['cpu']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_mem':
                x_data = [self.data['mem']]

            if Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'cpu':
                y_data = self.data['cpu']
            elif Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'mem':
                y_data = self.data['mem']
        else:
            print("We do not support your data for preprocessing!")

        x_timeseries = self.create_timeseries(x_data)
        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)
        valid_point = int((self.train_size + self.valid_size) * num_points)
        x_sampple = self.create_x(x_timeseries, sliding)
        x_train = x_sampple[0:train_point - sliding]
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (x_train.shape[0], sliding * int(x_train.shape[1])))

        x_valid = x_sampple[train_point - sliding: valid_point - sliding]
        x_valid = np.array(x_valid)
        x_valid = np.reshape(x_valid, (x_valid.shape[0], sliding * int(x_valid.shape[1])))

        x_test = x_sampple[valid_point - sliding:]
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], sliding * int(x_test.shape[1])))

        y_train = y_data[sliding: train_point]
        y_train = np.array(y_train)

        y_valid = y_data[train_point: valid_point]
        y_valid = np.array(y_valid)

        y_test = y_data[valid_point:]
        y_test = np.array(y_test)

        print('>>> Init data for training model done <<<')
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def create_x_decoder_from_x_encoder(self, x_encoder):
        x_decoder = []
        for i in range(len(x_encoder)):
            _x_decoder = [np.array([1] * x_encoder.shape[2])]
            for j in range(len(x_encoder[i]) - 1, 0, -1):
                _x_decoder.append(x_encoder[i][j])
            x_decoder.append(_x_decoder)
        return np.array(x_decoder)

    def create_y_decoder_from_x_encoder(self, x_encoder):
        y_decoder = []
        for i in range(len(x_encoder)):
            _y_decoder = []
            for j in range(len(x_encoder[i])):
                _y_decoder.append(x_encoder[i][j][0])
            _y_decoder.reverse()
            y_decoder.append(_y_decoder)
        y_decoder = np.array(y_decoder)
        y_decoder = np.reshape(y_decoder, (y_decoder.shape[0], 1, y_decoder.shape[1]))
        return y_decoder

    def init_data_autoencoder(self, sliding_encoder):
        print('>>> start init data for training autoencoder model <<<')

        if Config.DATA_EXPERIMENT == 'google_trace':
            if Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'cpu_mem':
                x_data = [self.data['cpu'], self.data['mem']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_cpu':
                x_data = [self.data['cpu']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_mem':
                x_data = [self.data['mem']]

            if Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'cpu':
                y_data = self.data['cpu']
            elif Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'mem':
                y_data = self.data['mem']
        else:
            print("We do not support your data for preprocessing!")

        x_timeseries = self.create_timeseries(x_data)
        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)
        valid_point = int((self.train_size + self.valid_size) * num_points)

        x_sampple_encoder = self.create_x(x_timeseries, sliding_encoder)

        x_train_encoder = x_sampple_encoder[0: train_point - sliding_encoder]
        x_train_encoder = np.array(x_train_encoder)
        x_train_decoder = self.create_x_decoder_from_x_encoder(x_train_encoder)
        y_train_decoder = self.create_y_decoder_from_x_encoder(x_train_encoder)

        x_valid_encoder = x_sampple_encoder[train_point - sliding_encoder: valid_point - sliding_encoder]
        x_valid_encoder = np.array(x_valid_encoder)
        x_valid_decoder = self.create_x_decoder_from_x_encoder(x_valid_encoder)
        y_valid_decoder = self.create_y_decoder_from_x_encoder(x_valid_encoder)

        x_test_encoder = x_sampple_encoder[valid_point - sliding_encoder:]
        x_test_encoder = np.array(x_test_encoder)
        x_test_decoder = self.create_x_decoder_from_x_encoder(x_test_encoder)
        y_test_decoder = self.create_y_decoder_from_x_encoder(x_test_encoder)
        print('>>> Init data for training autoencoder model complete <<<')

        return x_train_encoder, x_train_decoder, y_train_decoder, x_valid_encoder, x_valid_decoder, y_valid_decoder,\
            x_test_encoder, x_test_decoder, y_test_decoder

    def init_data_inference(self, sliding_encoder):
        print('>>> start init data for training inference model <<<')

        if Config.DATA_EXPERIMENT == 'google_trace':
            if Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'cpu_mem':
                x_data = [self.data['cpu'], self.data['mem']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_cpu':
                x_data = [self.data['cpu']]
            elif Config.GOOGLE_TRACE_DATA_CONFIG['train_data_type'] == 'uni_mem':
                x_data = [self.data['mem']]

            if Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'cpu':
                y_data = self.data['cpu']
            elif Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'mem':
                y_data = self.data['mem']
        else:
            print("We do not support your data for preprocessing!")

        x_timeseries = self.create_timeseries(x_data)
        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)
        valid_point = int((self.train_size + self.valid_size) * num_points)

        y_train_inference = y_data[sliding_encoder: train_point]
        y_train_inference = np.array(y_train_inference)

        y_valid_inference = y_data[train_point: valid_point]
        y_valid_inference = np.array(y_valid_inference)

        y_test_inference = y_data[valid_point:]
        y_test_inference = np.array(y_test_inference)
        print('>>> Init data for training BNN model complete <<<')

        return y_train_inference, y_valid_inference, y_test_inference
