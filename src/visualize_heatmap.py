import numpy as np
import matplotlib.pyplot as plt

import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.transforms as et


def heatmap(emg):
    # print(np.mean(emg))
    baseline = np.mean(emg[3])  # no motion
    max_90 = 0
    for g in emg:
        # max_90 = max(max_90, np.mean(g) + 0.75 * np.std(g))
        max_90 = max(max_90, np.percentile(g, 90))

        print(f"STD: {np.std(g):.2f}, Max: {max_90:.2f}")

    plt.figure()
    for gesture in range(emg.shape[0]):
        emg_g = emg[gesture]
        emg_g = emg_g.reshape(-1, 4, 16)
        emg_g = np.mean(emg_g, axis=0)

        plt.subplot(6, 1, gesture + 1)
        plt.imshow(emg_g, vmin=baseline, vmax=max_90)
        plt.ylabel(f"{gesture}")
        plt.xticks([])
        plt.yticks([])


if __name__ == "__main__":
    # Visualize heatmap of data.

    data_dir = "data/EMAGER/"

    SUBJECT = 1
    SESSION = 1

    # Compare session 1 and 2
    for i in range(2):
        data = ed.load_emager_data(data_dir, SUBJECT, i + 1)
        # emg = et.root_processing(data)
        # emg = et.default_processing(data)
        emg = np.abs(dp.filter_data(data))
        # if i == 0:
        # from emager_py.data_processing import _roll_array
        # emg = _roll_array(emg, -1)
        # emg = emg.reshape(6, 10, -1, 64)
        heatmap(emg)
        plt.title(f"{i}. Subject {SUBJECT} session {SESSION}")

    # Compare processing methods

    # data = ed.load_emager_data(data_dir, SUBJECT, SESSION)
    # for i in range(2):
    #     if i == 0:
    #         emg = np.abs(dp.filter_data(data))
    #         # emg = et.default_processing(data)
    #     else:
    #         emg = et.root_processing(data)
    #     heatmap(emg)
    #     plt.title(f"{i}. Subject {SUBJECT} session {SESSION}")

    # data = ed.load_emager_data(data_dir, SUBJECT, 1 if SESSION == 2 else 2)
    # heatmap(data)
    # plt.title(f"Subject {SUBJECT} session {1 if SESSION == 2 else 2}")

    plt.show()
