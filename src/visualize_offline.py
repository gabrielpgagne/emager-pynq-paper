import numpy as np
import matplotlib.pyplot as plt

import emager_py.dataset as ed
import emager_py.transforms as et
import globals as g

if __name__ == "__main__":
    SUBJECT = 0
    SESSION = 1

    GESTURE = 5
    REPETITION = 0

    data = ed.load_emager_data(g.EMAGER_DATASET_ROOT, SUBJECT, SESSION)
    data = et.default_processing(data)

    print("Loaded data with shape:", data.shape)

    data = data[GESTURE, REPETITION]
    data = np.array(data).reshape(-1, 64)
    t = np.arange(data.shape[0]) / g.EMAGER_SAMPLING_RATE
    for i in range(16):
        for j in range(4):
            plt.subplot(4, 16, 16 * j + i + 1)
            plt.plot(t, data[:, 16 * j + i])
    plt.show()
