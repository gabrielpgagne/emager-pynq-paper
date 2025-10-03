import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

import emager_py.dataset as ed
import emager_py.data_processing as dp


if __name__ == "__main__":
    # Visualize FFT before and after filtering.

    # Subject 9 session 1 repetition 0 seems to have motion artifact in gesture 0
    data_dir = "data/EMAGER/"

    SUBJECT = 0
    SESSION = 1

    GESTURES = [0]
    REPETITIONS = [0, 1, 2, 3, 4]

    # LOAD DATA
    data = ed.load_emager_data(data_dir, SUBJECT, SESSION)

    # data = data[GESTURES, REPETITIONS]
    data_processed = dp.filter_data(data)

    data = data.reshape(-1, 64)
    data_processed = data_processed.reshape(-1, 64)

    print("Loaded data with shape:", data.shape)

    print("Data stats:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Std: {np.std(data):.2f}")
    print(f"Max: {np.max(data):.2f}")
    print(f"Min: {np.min(data):.2f}")

    print("Processed data stats:")
    print(f"Mean: {np.mean(data_processed):.2f}")
    print(f"Std: {np.std(data_processed):.2f}")
    print(f"Max: {np.max(data_processed):.2f}")
    print(f"Min: {np.min(data_processed):.2f}")

    plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4 * j + i + 1)

            data0 = data[:, 16 * j + i]
            f = fft.rfftfreq(len(data0), 1 / 1000)
            y = fft.rfft(data0)
            plt.plot(f, 20 * np.log(np.abs(y)), alpha=0.7)

            data0 = data_processed[:, 16 * j + i]
            y = fft.rfft(data0)

            plt.plot(f, 20 * np.log(np.abs(y)), alpha=0.7)

    plt.show()
