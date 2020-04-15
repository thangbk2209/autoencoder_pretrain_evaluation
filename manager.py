import os
import tensorflow as tf
import time
import numpy as np
import pickle as pk
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import pandas as pd
import matplotlib

from config import *
from lib.model_training import ModelTrainer
from lib.data_visualization.read_grid_data import *
from lib.data_visualization.extract_data import *
from lib.data_visualization.visualize import *
matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


def read_data():
    google_trace_config = Config.GOOGLE_TRACE_DATA_CONFIG
    grid_data_config = Config.GRID_DATA_CONFIG
    traffic_data_config = Config.TRAFFIC_JAM_DATA_CONFIG
    normal_data_file = PROJECT_DIR + '/data/input_data/{}/normalized_data.pkl'.format(Config.DATA_EXPERIMENT)
    if Config.DATA_EXPERIMENT == 'google_trace':

        time_interval = google_trace_config['time_interval']
        data_name = google_trace_config['file_data_name'].format(time_interval)
        data_file_path = google_trace_config['data_path'].format(data_name)
        df = read_csv(data_file_path, header=None, index_col=False, names=google_trace_config['colnames'],
                      usecols=google_trace_config['usecols'], engine='python')
        cpu = df['cpu_rate'].values.reshape(-1, 1)
        mem = df['mem_usage'].values.reshape(-1, 1)
        disk_io_time = df['disk_io_time'].values.reshape(-1, 1)
        disk_space = df['disk_space'].values.reshape(-1, 1)

        if Config.VISUALIZATION_CONFIG['options']:
            official_data = {
                'cpu': cpu,
                'mem': mem,
                'disk_io_time': disk_io_time,
                'disk_space': disk_space
            }
            return official_data
        if not os.path.isfile(normal_data_file):
            # normalize data
            disk_io_time_scaler = MinMaxScaler(feature_range=(0, 1))
            disk_io_time_normal = disk_io_time_scaler.fit_transform(disk_io_time)

            disk_space_scaler = MinMaxScaler(feature_range=(0, 1))
            disk_space_normal = disk_space_scaler.fit_transform(disk_space)

            mem_scaler = MinMaxScaler(feature_range=(0, 1))
            mem_normal = mem_scaler.fit_transform(mem)

            cpu_scaler = MinMaxScaler(feature_range=(0, 1))
            cpu_normal = cpu_scaler.fit_transform(cpu)

            normalized_data = {
                'cpu': cpu_normal,
                'mem': mem_normal,
                'disk_io_time': disk_io_time_normal,
                'disk_space': disk_space_normal
            }

            with open(normal_data_file, 'wb') as normal_data_file:
                pk.dump(cpu_scaler, normal_data_file, pk.HIGHEST_PROTOCOL)
                pk.dump(mem_scaler, normal_data_file, pk.HIGHEST_PROTOCOL)
                pk.dump(disk_io_time_scaler, normal_data_file, pk.HIGHEST_PROTOCOL)
                pk.dump(disk_space_scaler, normal_data_file, pk.HIGHEST_PROTOCOL)
                pk.dump(normalized_data, normal_data_file, pk.HIGHEST_PROTOCOL)
        else:
            with open(normal_data_file, 'rb') as normal_data_file:
                cpu_scaler = pk.load(normal_data_file)
                mem_scaler = pk.load(normal_data_file)
                disk_io_time_scaler = pk.load(normal_data_file)
                disk_space_scaler = pk.load(normal_data_file)
                normalized_data = pk.load(normal_data_file)
        if Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'cpu':
            return normalized_data, cpu_scaler
        elif Config.GOOGLE_TRACE_DATA_CONFIG['predict_data'] == 'mem':
            return normalized_data, mem_scaler
        else:
            print('>>> This prediction data is not served <<<')
            return None, None
    elif Config.DATA_EXPERIMENT == 'grid':
        time_interval = grid_data_config['time_interval']
        file_data_name = grid_data_config['file_data_name'].format(time_interval / 60)
        data_file_path = grid_data_config['data_path'].format(file_data_name)
        colnames = grid_data_config['colnames']
        df = read_csv(data_file_path, header=None, index_col=False, names=grid_data_config['colnames'],
                      engine='python')
        jobs_id = df['job_id_data'].values.reshape(-1, 1)
        n_processes = df['n_proc_data'].values.reshape(-1, 1)
        used_cpu_time = df['used_cpu_time_data'].values.reshape(-1, 1)
        used_memory = df['used_memory_data'].values.reshape(-1, 1)
        users_id = df['user_id_data'].values.reshape(-1, 1)
        groups_id = df['group_id_data'].values.reshape(-1, 1)
        n_processes = n_processes[384:]
        n_processes_scaler = MinMaxScaler(feature_range=(0, 1))
        n_processes_normal = n_processes_scaler.fit_transform(n_processes)

        if Config.VISUALIZATION_CONFIG['options']:
            official_data = {
                'jobs_id': jobs_id,
                'n_processes': n_processes,
                'used_cpu_time': used_cpu_time,
                'used_memory': used_memory,
                'users_id': users_id,
                'groups_id': groups_id
            }
            return official_data
        return n_processes_normal, n_processes_scaler
    elif Config.DATA_EXPERIMENT == 'traffic':
        file_data_name = traffic_data_config['file_data_name']
        colnames = traffic_data_config['colnames']
        file_data_path = traffic_data_config['data_path'].format(file_data_name)

        df = read_csv(file_data_path, header=None, index_col=False, names=colnames, engine='python')
        traffic = df['megabyte'].values.reshape(-1, 1)

        if Config.VISUALIZATION_CONFIG['options']:
            official_data = {
                'eu': traffic
            }
            return official_data

        traffic_scaler = MinMaxScaler(feature_range=(0, 1))
        traffic_normal = traffic_scaler.fit_transform(traffic)

        return traffic_normal, traffic_scaler
    else:
        print('>>> We do not support to experiment with this data <<<')
        return None, None


def init_model():
    print('[1] >>> Start init model')
    normalized_data, scaler = read_data()
    model_trainer = ModelTrainer(normalized_data, scaler)
    model_trainer.train()
    print('[1] >>> Init model complete')


def visualize_data():
    official_data = read_data()
    data2visualize = official_data[Config.VISUALIZATION_CONFIG['metrics'][Config.DATA_EXPERIMENT]]
    print('>>> Start visualization <<<')
    visualize(data2visualize, Config.DATA_EXPERIMENT, Config.VISUALIZATION_CONFIG['metrics'][Config.DATA_EXPERIMENT])
    print('>>> Visualize complete <<<')


def plot_results():
    normalized_data, scaler = read_data()
    whale_folder_path = '/Users/thangnguyen/hust_project/cloud_autoscaling/data/lstm/whale/traffic/results/'
    whale_file_name = ['sli-2_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-3_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-4_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-5_batch-8_numunits-4_act-tanh_opt_adam_num_par-50']

    bp_folder_path = '/Users/thangnguyen/hust_project/cloud_autoscaling/data/lstm/bp/traffic/results/'
    bp_file_name = ['sli-5_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-2_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-3_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-4_batch-8_numunits-4_act-tanh_opt_adam']
    folder_path = bp_folder_path
    file_name = bp_file_name
    for _file_name in file_name:
        file_path = '{}{}.csv'.format(folder_path, _file_name)
        df = pd.read_csv(file_path)
        pred_data = df.values
        print(pred_data.shape)
        real_data = scaler.inverse_transform(normalized_data)
        real_data = real_data[-pred_data.shape[0]:]
        ax = plt.subplot()
        ax.plot(real_data, label="Actual")
        ax.plot(pred_data, label="predictions")
        plt.xlabel("TimeStamp")
        plt.ylabel("Traffic")
        plt.legend()
        # plt.show()
        plt.savefig('{}{}.png'.format(folder_path, _file_name))
        plt.close()


if __name__ == "__main__":
    print('start')
    if Config.VISUALIZATION_CONFIG['options']:
        visualize_data()
    else:
        init_model()
    # plot_results()
