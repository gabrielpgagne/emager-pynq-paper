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
        sns.pointplot(r, x="shots", y=metric.value, linewidth=2.5)


def plot_accuracy_vs_quant_per_subject(base_dir, shots, metric: utils.ModelMetric):
    """
    For every quantization, plot the accuracy across all subjects for a given `shots` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)
    print(results)

    for r in results:
        sns.pointplot(r, x="quantization", y=metric.value, linewidth=2.5)

def accuracy_per_subject(base_dir, shots, quant, metric: utils.ModelMetric):
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)
    ret = []
    for i, r in enumerate(results):
        v = r.loc[r['quantization'] == quant][metric.value].values[0]
        ret.append(v)
        tshots = f"{shots:02d}" if shots != -1 else "all"
        print(f"Subject {i:02d} accuracy for {tshots} shots and {quant:02d}-bit quant: {v*100:0.1f}%")
    return ret

if __name__ == "__main__":
    # TODO: add axis labels, add legend, cleanup figures
    import globals as g

    accuracy_per_subject(g.OUT_DIR_STATS, 10, 8, utils.ModelMetric.ACC_MAJ)
    sns.set_theme(style="darkgrid")

    plt.figure(1)
    plot_accuracy_vs_quant_per_subject(g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_MAJ)
    plt.savefig(g.OUT_DIR_ROOT + "img/accuracy_vs_quant.png", bbox_inches='tight')
    plt.figure(2)
    plt.subplot(1, 3, 1)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_MAJ)
    plt.subplot(1, 3, 2)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_STATS, 8, utils.ModelMetric.ACC_MAJ)
    plt.subplot(1, 3, 3)
    plot_accuracy_vs_shots_per_subject(g.OUT_DIR_STATS, 3, utils.ModelMetric.ACC_MAJ)
    plt.savefig(g.OUT_DIR_ROOT + "img/accuracy_vs_shots.png")
