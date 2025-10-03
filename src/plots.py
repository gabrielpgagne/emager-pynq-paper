import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.quantization as eq

import globals as g
import utils


def raw_acc_vs_maj_acc(base_dir, shots, quants):
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)

    acc_pairs_intra = np.zeros((len(results), 2))
    acc_pairs_inter = np.zeros((len(results), 2))
    for q in quants:
        for i, r in enumerate(results):
            line = r.loc[r["quantization"] == q]
            acc_raw, acc_maj = (
                line[utils.ModelMetric.ACC_RAW_INTRA.value].values[0],
                line[utils.ModelMetric.ACC_MAJ_INTRA.value].values[0],
            )
            acc_pairs_intra[i] = [acc_raw, acc_maj]
            acc_raw, acc_maj = (
                line[utils.ModelMetric.ACC_RAW_INTER.value].values[0],
                line[utils.ModelMetric.ACC_MAJ_INTER.value].values[0],
            )
            acc_pairs_inter[i] = [acc_raw, acc_maj]

        print(
            ("\\multirow{2}{*}{%s} & Intra & " % q)
            + f"$ {100 * np.mean(acc_pairs_intra[:, 0:1], axis=0).item():.1f} \\pm {100 * np.std(acc_pairs_intra[:, 0:1], axis=0).item():.1f} $ & $ {100 * np.mean(acc_pairs_intra[:, 1:], axis=0).item():.1f} \\pm {100 * np.std(acc_pairs_intra[:, 1:], axis=0).item():.1f} $"
            + r" \\ \cline{2-2}"
        )
        print(
            "                   & Inter & "
            + f"$ {100 * np.mean(acc_pairs_inter[:, 0:1], axis=0).item():.1f} \\pm {100 * np.std(acc_pairs_inter[:, 0:1], axis=0).item():.1f} $ & $ {100 * np.mean(acc_pairs_inter[:, 1:], axis=0).item():.1f} \\pm {100 * np.std(acc_pairs_inter[:, 1:], axis=0).item():.1f} $"
            + r" \\ \cline{1-4}"
        )
    return acc_pairs_intra, acc_pairs_inter


def plot_accuracy_vs_shots_per_subject(
    base_dir, quantization, metric: utils.ModelMetric
):
    """
    For every subject, plot the accuracy across all cross-validations for a given `quantization` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_shots(
        base_dir, quantization
    )
    print(results)
    styles = ["solid", "dotted", "dashed"]
    for i, r in enumerate(results):
        r[metric.value] = r[metric.value] * 100
        r = r[r["shots"] != -1]
        sns.pointplot(
            r,
            x="shots",
            y=metric.value,
            linewidth=4.0,
            linestyles=styles[i // 10],
            label=f"Subject {i}",
            legend=False,
        )

    sns.despine()
    plt.ylabel("Majority Vote Accuracy [%]")


def plot_accuracy_vs_quant_per_subject(base_dir, shots, metric: utils.ModelMetric):
    """
    For every quantization, plot the accuracy across all subjects for a given `shots` and `metric`.
    """
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)
    print(results)
    styles = ["solid", "dotted", "dashed"]
    for i, r in enumerate(results):
        r[metric.value] = r[metric.value] * 100
        sns.pointplot(
            r,
            x="quantization",
            y=metric.value,
            linewidth=4.0,
            linestyles=styles[i // 10],
            label=f"Subject {i}",
        )

    sns.despine()
    plt.ylabel("Majority Vote Accuracy [%]")
    plt.xlabel("Quantization bits")


def accuracy_per_subject(base_dir, shots, quant, metric: utils.ModelMetric):
    results: list[pd.DataFrame] = utils.get_all_accuracy_vs_quant(base_dir, shots)
    ret = []
    for i, r in enumerate(results):
        v = r.loc[r["quantization"] == quant][metric.value].values[0]
        ret.append(v)
        tshots = f"{shots:02d}" if shots != -1 else "all"
        print(
            f"Subject {i:02d} accuracy for {tshots} shots and {quant:02d}-bit quant: {v * 100:0.1f}%"
        )
    return ret


def concat_gestures(dataset_root, subject, session, rep, channel):
    ses_data = ed.load_emager_data(dataset_root, subject, session)
    data = np.zeros((0,))
    data = np.concatenate((data, ses_data[0, rep, :1000, channel]))  # power grip
    data = np.concatenate((data, ses_data[1, rep, :1000, channel]))
    data = np.concatenate((data, ses_data[2, rep, :1000, channel]))
    data = np.concatenate((data, ses_data[3, rep, :1000, channel]))
    data = np.concatenate((data, ses_data[4, rep, :1000, channel]))
    data = np.concatenate((data, ses_data[5, rep, :1000, channel]))
    data = data.reshape(-1, 1)
    # data = data - np.mean(data)
    # data = data * 0.195 * 10e-3
    return data


def plot_rep(dataset_root, subject, session, rep, channel):
    data = concat_gestures(dataset_root, subject, session, rep, channel)
    time = np.linspace(0, len(data) / g.EMAGER_SAMPLING_RATE, len(data)).reshape(-1, 1)
    ndf = np.hstack((data, time))
    df = pd.DataFrame(ndf, columns=["EMG", "Time [s]"])
    ax = sns.lineplot(df, x="Time [s]", y="EMG")
    sns.despine()
    plt.ylabel("EMG$_{int16}$")
    return ax


def plot_processed(dataset_root, subject, session, rep, channel):
    data = concat_gestures(dataset_root, subject, session, rep, channel)
    datap = dp.preprocess_data(data)
    datap = eq.nroot_c(datap, 1.7, 8)
    timep = np.linspace(
        0, 25 * len(datap) / g.EMAGER_SAMPLING_RATE, len(datap)
    ).reshape(-1, 1)
    ndf = np.hstack((datap, timep))
    df = pd.DataFrame(ndf, columns=["EMG", "Time [s]"])
    ax = sns.lineplot(df, x="Time [s]", y="EMG", linewidth=3.0)
    sns.despine()
    plt.ylabel("EMG$_{int16}$")
    return ax


def plot_quantization():
    data = np.arange(2**15).reshape(-1, 1)
    qdata = eq.nroot_c(data, 1.7, 8)
    ndf = np.hstack((data, qdata))
    df = pd.DataFrame(ndf, columns=["int16", "uint8"])
    ax = sns.lineplot(df, x="int16", y="uint8", linewidth=4)
    sns.despine()
    return ax


def plot_pixelmap(dataset_root, subject, session, rep):
    data = ed.load_emager_data(dataset_root, subject, session)
    data = dp.filter_data(data)
    image = data[0, rep, 12, :].reshape((4, 16))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image)
    plt.axis("off")


def get_resources_vs_quant(quants, shots):
    vals = [[] for _ in range(len(quants))]
    vals_rel = [[] for _ in range(len(quants))]
    for subject in os.listdir(g.OUT_DIR_ROOT + g.OUT_DIR_FINN):
        try:
            subject = int(subject)
        except:  # noqa
            continue
        for i, quant in enumerate(quants):
            try:
                val = utils.get_accelerator_resources(subject, quant, shots)
            except Exception as e:
                print(e)
                continue
            vals[i].append(val)
            vals_rel[i].append(100 * val / 53200)
    # print(vals)
    # print([np.mean(v) for v in vals])
    # print([np.std(v) for v in vals])

    df = pd.DataFrame(
        {
            "Quantization": quants,
            "LUTs": [np.mean(v) for v in vals],
            "LUTs STD": [np.std(v) for v in vals],
            "LUTs [%]": [np.mean(v) for v in vals_rel],
            "LUTs STD [%]": [np.std(v) for v in vals_rel],
        }
    )
    ax1 = sns.pointplot(df, x="Quantization", y="LUTs")
    plt.ylabel("Total LUTs used")
    plt.xlabel("Quantization bits")
    plt.grid(True, "both", "both")

    ax2_lims = [100 * l / 53200 for l in ax1.get_ylim()]
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total LUTs used [%]")
    ax2.set_ylim(ax2_lims[0], ax2_lims[1])
    sns.despine(right=False)
    return df


if __name__ == "__main__":
    # sns.set_theme(font_scale=3)

    font = {"size": 36}
    matplotlib.rc("font", **font)

    # ==========================================================
    # EMG example
    # ==========================================================

    # plt.figure()
    # plot_pixelmap(g.EMAGER_DATASET_ROOT, 0, 1, 0)
    # plt.savefig(g.OUT_DIR_ROOT + "img/pixelmap.png", bbox_inches="tight")
    # exit()

    # ==========================================================
    # Quantization operation
    # ==========================================================

    # ax = plot_quantization()
    # plt.grid(True, "both", "both")
    # plt.savefig(g.OUT_DIR_ROOT + "img/quantization.png", format="png", bbox_inches="tight")

    # ==========================================================
    # Raw vs processed EMG
    # ==========================================================

    # plt.figure()
    # plt.rcParams.update({'font.size': 22})
    # plt.figure()
    # plot_rep(g.EMAGER_DATASET_ROOT, 0, 1, 0, 0)
    # plt.savefig(g.OUT_DIR_ROOT + "img/emg_unprocessed.png", bbox_inches="tight")
    # plt.figure()
    # plot_processed(g.EMAGER_DATASET_ROOT, 0, 1, 0, 0)
    # plt.savefig(g.OUT_DIR_ROOT + "img/emg_processed.png", bbox_inches="tight")

    # ==========================================================
    # Print accuracy per subject
    # ==========================================================

    # accuracy_per_subject(g.OUT_DIR_STATS, -1, 8, utils.ModelMetric.ACC_MAJ_INTER)

    # ==========================================================
    # Accuracy VS quantization, per subject
    # ==========================================================

    figsize = (32, 16)
    plt.figure(1, figsize=figsize)
    # plt.subplot(4,1,1)
    plt.grid(True, "both", "both")
    plt.ylim(0, 100)
    plot_accuracy_vs_quant_per_subject(
        g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_MAJ_INTER
    )
    plt.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.2, 1.0))
    # plt.show()
    plt.savefig(g.OUT_DIR_ROOT + "img/accuracy_vs_quant.png", bbox_inches="tight")

    # ==========================================================
    # Accuracy VS shots, per subject, no quantization
    # ==========================================================

    figsize = (32, 40)
    plt.figure(2, figsize=figsize)

    plt.subplot(3, 1, 1)
    plot_accuracy_vs_shots_per_subject(
        g.OUT_DIR_STATS, -1, utils.ModelMetric.ACC_MAJ_INTER
    )
    plt.ylim(40, 100)
    plt.grid(True, "both", "both")
    plt.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.2, 1.0))
    plt.xlabel("(a)")

    # ==========================================================
    # Accuracy VS shots, per subject, 8-bit quantization
    # ==========================================================

    # plt.figure(3, figsize=figsize)
    plt.subplot(3, 1, 2)
    plot_accuracy_vs_shots_per_subject(
        g.OUT_DIR_STATS, 8, utils.ModelMetric.ACC_MAJ_INTER
    )
    plt.ylim(40, 100)
    plt.grid(True, "both", "both")
    plt.xlabel("(b)")

    # ==========================================================
    # Accuracy VS shots, per subject, 4-bit quantization
    # ==========================================================

    # plt.figure(4, figsize=figsize)
    plt.subplot(3, 1, 3)
    plot_accuracy_vs_shots_per_subject(
        g.OUT_DIR_STATS, 4, utils.ModelMetric.ACC_MAJ_INTER
    )
    plt.ylim(40, 100)
    plt.grid(True, "both", "both")
    plt.xlabel("(c)\nNumber of shots")

    plt.savefig(g.OUT_DIR_ROOT + "img/accuracy_vs_shots.png", bbox_inches="tight")

    # ==========================================================
    #
    # ==========================================================

    raw_acc_vs_maj_acc(g.OUT_DIR_STATS, -1, [1, 2, 3, 4, 6, 8, 32])

    # ==========================================================
    # Hardware resources VS quant
    # ==========================================================
    #
    # plt.figure(10, figsize=(24, 12))
    # qr = get_resources_vs_quant([2, 3, 4, 6, 8], 20)
    # print(qr.to_latex())
    # plt.savefig(g.OUT_DIR_ROOT + "img/luts_vs_quantization.png", bbox_inches="tight")
    # plt.show()
