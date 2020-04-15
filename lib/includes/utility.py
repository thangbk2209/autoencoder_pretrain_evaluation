"""
  Author:  thangbk2209
  Project: Autoscaling
  Created: 3/15/19 16:48
  Purpose:
"""
import matplotlib
from config import *
matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


def draw_time_series(data, title, x_label, y_label, file_name):
    plt.plot(data)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.legend([/], loc='upper left')
    plt.savefig(file_name + '.png')
    plt.show()
    plt.close()

