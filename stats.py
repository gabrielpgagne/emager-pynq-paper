import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import emager_py.dataset as ed

import globals
import utils
import os


def plot_shots_accuracy_per_subject(quantization, metric: utils.ModelMetric):
    """
    For every subject, plot the accuracy across all cross-validations for a given `quantization` and `metric`.
    """
    for results in utils.get_all_statistics(quantization, metric):
        shots_acc = {"shots": [], "accuracy": [], "std": []}
        for k, v in results.items():
            if k == "-1" or not k.isnumeric():
                continue
            shots_acc["shots"].append(int(k))
            shots_acc["accuracy"].append(v["acc_avg"])
            shots_acc["std"].append(v["acc_std"])
        data = pd.DataFrame(shots_acc)
        sns.lineplot(data, x="shots", y="accuracy", linewidth=2.5)


# def plot_accuracy_per_quant():

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    plot_shots_accuracy_per_subject(32, utils.ModelMetric.ACC_RAW)

    plt.show()
