import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import utils


def plot_accuracy_vs_shots_per_subject(quantization, metric: utils.ModelMetric):
    """
    For every subject, plot the accuracy across all cross-validations for a given `quantization` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_shots(quantization)
    for r in results:
        r = r[r["shots"] != -1]
        sns.lineplot(r, x="shots", y=metric.value, linewidth=2.5)


def plot_accuracy_vs_quant_per_subject(shots, metric: utils.ModelMetric):
    """
    For every quantization, plot the accuracy across all subjects for a given `shots` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(shots)
    print(results[0])
    for r in results:
        sns.lineplot(r, x="quantization", y=metric.value, linewidth=2.5)


if __name__ == "__main__":
    # TODO: add axis labels, add legend, cleanup figures

    sns.set_theme(style="darkgrid")

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plot_accuracy_vs_quant_per_subject(-1, utils.ModelMetric.ACC_RAW)
    plt.subplot(1, 2, 2)
    plot_accuracy_vs_quant_per_subject(10, utils.ModelMetric.ACC_RAW)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plot_accuracy_vs_shots_per_subject(1, utils.ModelMetric.ACC_RAW)
    plt.subplot(1, 2, 2)
    plot_accuracy_vs_shots_per_subject(3, utils.ModelMetric.ACC_RAW)
    plt.show()
