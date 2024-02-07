import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve", shift = 0):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y[:25], window=5)
    # Truncate x
    # x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    label = log_folder.split("-")
    if label[-1] == 'v0':
        label = 'No-eploration'
    else: label = label[-2] + '-freq-' + label[-1]
    plt.plot(y, label=label)
    plt.xlabel("episode")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")



plot_results("ppo-default-params/MountainCarContinuous-v0-TreeCounterCV-16384")
plot_results("ppo-default-params/MountainCarContinuous-v0-TreeCounterCVWSOnly-16384")
plot_results("ppo-default-params/MountainCarContinuous-v0-TreeCounter-16384")
plot_results("ppo-default-params/MountainCarContinuous-v0-TreeCounterWSOnly-16384")
plot_results("ppo-default-params/MountainCarContinuous-v0")
plot_results('ppo-default-params/MountainCarContinuous-v0-ForestCounter-2048')
plt.grid()
plt.legend()

plt.savefig("MountainCar.pdf")
