import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')


class Config:
    DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace
    PLT_ENV = 'Agg'  # TkAgg
    GOOGLE_TRACE_DATA_CONFIG = {
        'train_data_type': 'cpu_mem',  # cpu_mem, uni_mem, uni_cpu
        'predict_data': 'cpu',
        'time_interval': 5,
        'file_data_name': '/input_data/google_trace/data_{}Min.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['cpu_rate', 'mem_usage', 'disk_io_time', 'disk_space'],
        'usecols': [3, 4, 9, 10]
    }
    GRID_DATA_CONFIG = {
        'time_interval': 10800,  # 600, 3600, 7200, 10800, 21600
        'file_data_name': '/input_data/grid_data/timeseries_anonjobs_{}Min.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['job_id_data', 'n_proc_data', 'used_cpu_time_data', 'used_memory_data', 'user_id_data',
                     'group_id_data']
    }
    TRAFFIC_JAM_DATA_CONFIG = {
        'file_data_name': '/input_data/traffic/it_eu_5m.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['timestamp', 'bit', 'byte', 'kilobyte', 'megabyte']
    }
    VISUALIZATION_CONFIG = {
        'options': False,
        'metrics': {
            'google_trace': 'disk_space',
            'grid': 'n_processes',  # 'n_processes', 'used_cpu_time', 'used_memory', 'users_id', 'groups_id'
            'traffic': 'eu'
        }
    }
    VISUALIZATION = False
    MODEL_EXPERIMENT = 'lstm'  # lstm, ann, bnn, pretrain_autoencoder
    METHOD_APPROACH = 'bp'  # pso, whale, bp, bp_pso, pso_bp

    LEARNING_RATE = 3e-4
    EPOCHS = 10
    EARLY_STOPPING = True
    PATIENCE = 20
    TRAIN_SIZE = 0.6
    VALID_SIZE = 0.2

    if DATA_EXPERIMENT == 'google_trace':
        INFO_PATH = '{}/{}/{}/{}'.format(
            MODEL_EXPERIMENT, METHOD_APPROACH, GOOGLE_TRACE_DATA_CONFIG['train_data_type'],
            GOOGLE_TRACE_DATA_CONFIG['predict_data'])
        MODEL_SAVE_PATH = CORE_DATA_DIR + '/{}/model/'.format(INFO_PATH)
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/results/'.format(INFO_PATH)
        TRAIN_LOSS_PATH = CORE_DATA_DIR + '/{}/train_losses/'.format(INFO_PATH)
        EVALUATION_PATH = CORE_DATA_DIR + '/{}/evaluation.csv'.format(INFO_PATH)
    else:
        INFO_PATH = '{}/{}/{}'.format(MODEL_EXPERIMENT, METHOD_APPROACH, DATA_EXPERIMENT)
        MODEL_SAVE_PATH = CORE_DATA_DIR + '/{}/model/'.format(INFO_PATH)
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/results/'.format(INFO_PATH)
        TRAIN_LOSS_PATH = CORE_DATA_DIR + '/{}/train_losses/'.format(INFO_PATH)
        EVALUATION_PATH = CORE_DATA_DIR + '/{}/evaluation.csv'.format(INFO_PATH)

    LSTM_CONFIG = {
        'time_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'sliding': [12],
        'batch_size': [64, 128],
        'num_units': [[4], [4, 2], [8, 4, 2], [16, 8, 4, 2]],
        'dropout_rate': [0.9],
        'variation_dropout': False,
        'activation': ['tanh'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['adam'],  # 'momentum', 'adam', 'rmsprop'
    }

    ANN_CONFIG = {
        'sliding': [3],
        'batch_size': [8],
        'num_units': [[4]],
        'activation': ['sigmoid'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['momentum']  # 'momentum', 'adam', 'rmsprop'
    }

    AUTOENCODER_PRETRAIN_CONFIG = {
        'time_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'sliding_encoder': [12],
        'batch_size': [64, 128],
        'num_units_lstm': [[4], [4, 2], [8, 4, 2], [16, 8, 4, 2]],
        'num_units_inference': [[4]],
        'dropout_rate': [0.9],
        'variation_dropout': False,
        'activation': ['tanh'],
        'optimizer': ['adam'],
        'variant': [False]
    }

# 'sliding_encoder': [10, 12],
# 'sliding_inference': [2, 3, 4, 5],
# 'batch_size': [8, 32, 128],
# 'num_units_lstm': [[32, 4], [8]],
# 'num_units_inference': [[4], [16, 4]],
# 'dropout_rate': [0.9],
# 'variation_dropout': False,
# 'activation': ['relu', 'elu'],
# 'optimizer': ['momentum', 'adam', 'rmsprop']
