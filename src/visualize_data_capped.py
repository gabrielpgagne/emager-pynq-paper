import numpy as np
import matplotlib.pyplot as plt

import emager_py.dataset as ed
import emager_py.data_processing as dp


if __name__ == "__main__":
    # Visualize analog data after filtering with a cap.

    # Subject 9 session 1 repetition 0 seems to have motion artifact in gesture 0
    data_dir = "data/EMAGER/"

    SUBJECT = 11

    GESTURES = [i for i in range(6)]
    REPETITIONS = [2]

    cap = 500  # uV

    # LOAD DATA
    plt.figure()
    for i in range(2):
        data = ed.load_emager_data(data_dir, SUBJECT, i + 1)
        data = data * 0.195  # uV
        data = data[GESTURES, REPETITIONS]

        data_processed = dp.filter_data(data)
        # data_processed = data
        data_processed = data_processed.reshape(-1, 64)
        data_processed = np.clip(data_processed, -cap, cap)

        print("Loaded data with shape:", data.shape)

        print("Processed data stats:")
        print(f"Mean: {np.mean(data_processed):.2f}")
        print(f"Std: {np.std(data_processed):.2f}")
        print(f"Max: {np.max(data_processed):.2f}")
        print(f"Min: {np.min(data_processed):.2f}")

        t = np.arange(data_processed.shape[0]) / 1000
        for i in range(16):
            for j in range(4):
                plt.subplot(4, 16, 16 * j + i + 1)
                plt.plot(t, data_processed[:, 16 * j + i], alpha=0.7)
                plt.ylim(-500, 500)

    plt.show()
