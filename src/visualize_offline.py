import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

import emager_py.dataset as ed
import emager_py.transforms as et
import globals as g


def heatmap(emg):
    plt.figure()
    emg = et.default_processing(emg)

    for gesture in range(emg.shape[0]):
        emg_g = emg[gesture : gesture + 1]
        emg_g = emg_g.reshape(-1, 4, 16)
        emg_g = np.mean(emg_g, axis=0)

        print(
            f"({gesture}) mean {np.mean(emg_g)}, std {np.std(emg_g)}, max {np.max(emg_g)}, min {np.min(emg_g)}"
        )
        plt.subplot(6, 1, gesture + 1)
        plt.imshow(emg_g, vmin=0, vmax=200)
        plt.ylabel(f"{gesture}")
        plt.xticks([])
        plt.yticks([])
    # plt.colorbar()
    plt.show()


if __name__ == "__main__":
    SUBJECT = 1
    SESSION = 2

    GESTURE = 2
    REPETITION = 0

    data = ed.load_emager_data(g.EMAGER_DATASET_ROOT, SUBJECT, SESSION)
    # data = et.default_processing(data)

    # heatmap(data)

    data = data[GESTURE : GESTURE + 1]

    print("Loaded data with shape:", data.shape)

    # data = data[GESTURE, REPETITION]
    data = np.array(data).reshape(-1, 64)
    t = np.arange(data.shape[0]) / g.EMAGER_SAMPLING_RATE
    for i in range(16):
        for j in range(4):
            plt.subplot(4, 16, 16 * j + i + 1)
            plt.plot(t, data[:, 16 * j + i])

    for i in range(1):
        data0 = data[:, i]
        y = fft.fft(data0)[: len(data0) // 2]
        f = fft.fftfreq(len(data0), 1 / 1000)[: len(data0) // 2]
        plt.figure()
        plt.plot(f, 20 * np.log(np.abs(y)))

    plt.show()
