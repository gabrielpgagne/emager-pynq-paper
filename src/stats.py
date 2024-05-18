import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import utils


def plot_accuracy_vs_shots_per_subject(base_dir, quantization, metric: utils.ModelMetric):
    """
    For every subject, plot the accuracy across all cross-validations for a given `quantization` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_shots(base_dir, quantization)
    print(results)
    for r in results:
        r = r[r["shots"] != -1]
        sns.lineplot(r, x="shots", y=metric.value, linewidth=2.5)


def plot_accuracy_vs_quant_per_subject(base_dir, shots, metric: utils.ModelMetric):
    """
    For every quantization, plot the accuracy across all subjects for a given `shots` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)
    print(results)

    for r in results:
        sns.lineplot(r, x="quantization", y=metric.value, linewidth=2.5)


if __name__ == "__main__":
    # TODO: add axis labels, add legend, cleanup figures
    import globals as g

    sns.set_theme(style="darkgrid")

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plot_accuracy_vs_quant_per_subject(g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_RAW)
    plot_accuracy_vs_quant_per_subject(g.OUT_DIR_FINN, 10, utils.ModelMetric.ACC_RAW)
    plt.subplot(1, 2, 2)
    #plot_accuracy_vs_quant_per_subject(g.OUT_DIR_STATS, 10, utils.ModelMetric.ACC_RAW)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_RAW)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_STATS, 3, utils.ModelMetric.ACC_RAW)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_FINN, 8, utils.ModelMetric.ACC_RAW)
    plt.subplot(1, 2, 2)
    #plt.show()
