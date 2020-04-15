"""
  Author:  thangbk2209
  Project: Autoscaling
  Created: 3/15/19 11:00
  Purpose:
"""
import matplotlib

from config import *
from lib.includes.utility import *

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


def visualize(data, data_experiment, metric):
    file_name = PROJECT_DIR + '/{}/{}/{}/{}'.format('data', 'visualization', data_experiment, metric)
    title = data_experiment
    x_label = 'Time stamp'
    y_label = metric
    draw_time_series(data, title, x_label, y_label, file_name)
