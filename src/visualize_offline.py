import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

import emager_py.dataset as ed
import emager_py.data_processing as dp

import utils
import globals as g


def heatmap(emg):
    plt.figure()
    emg = dp.preprocess_data(emg)

    baseline = utils.noise_floor(emg) / 2
    max = np.percentile(emg[0], 90)  # Power grip

    # print(f"Baseline: {baseline}, Max: {max}")

    for gesture in range(emg.shape[0]):
        emg_g = emg[gesture : gesture + 1]
        emg_g = emg_g.reshape(-1, 4, 16)
        emg_g = np.mean(emg_g, axis=0)

        plt.subplot(6, 1, gesture + 1)
        plt.imshow(emg_g, vmin=baseline, vmax=max)
        plt.ylabel(f"{gesture}")
        plt.xticks([])
        plt.yticks([])


if __name__ == "__main__":
    SUBJECT = 14
    SESSION = 1

    GESTURE = 0
    REPETITION = 0

    data_dir = "data/live_test/"

    data = ed.load_emager_data(data_dir, SUBJECT, SESSION)

    noise = utils.noise_floor(data)
    print(f"RMS Noise floor: {noise:.2f}")
    heatmap(data)
    plt.title(f"Subject {SUBJECT}")
    # plt.show()

    # exit()
    data = data[GESTURE : GESTURE + 1, REPETITION : REPETITION + 1]

    print("Loaded data with shape:", data.shape)

    # data = data[GESTURE, REPETITION]
    data = np.array(data).reshape(-1, 64)
    t = np.arange(data.shape[0]) / g.EMAGER_SAMPLING_RATE
    plt.figure(5)
    for i in range(16):
        for j in range(4):
            plt.subplot(4, 16, 16 * j + i + 1)
            plt.plot(t, data[:, 16 * j + i])

    for i in range(1):
        data0 = data[:, i]
        y = fft.fft(data0)[: len(data0) // 2]
        f = fft.fftfreq(len(data0), 1 / 1000)[: len(data0) // 2]
        plt.figure(6)
        plt.plot(f, 20 * np.log(np.abs(y)))

    plt.show()
