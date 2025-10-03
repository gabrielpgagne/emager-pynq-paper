import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.transforms as etrans


def main():
    # Visualize EMG data before and after transforms.

    data_dir = "data/EMAGER/"

    SUBJECT = 0
    SESSION = 1
    REPETITION = [0]

    # LOAD
    emg_data = ed.load_emager_data(data_dir, SUBJECT, SESSION)
    emg_data = emg_data[:, REPETITION, ...]

    # Process the EMG data
    # processed_data = etrans.root_processing(emg_data)
    processed_data = etrans.default_processing(emg_data)

    # Normalize
    emg_data = (emg_data - np.min(emg_data)) / (np.max(emg_data) - np.min(emg_data))
    processed_data = processed_data / 255

    # Reshape the data to 2D (samples, channels)
    emg_data = emg_data.reshape(-1, 64)
    processed_data = processed_data.reshape(-1, 64)

    time = np.arange(len(emg_data)) / 1000  # Time in seconds

    # Get the actual number of channels (up to 16)
    n_channels = min(16, emg_data.shape[-1])

    # Create a 4x4 grid for up to 16 channels
    rows = 4
    cols = n_channels // rows + (n_channels % rows > 0)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot each channel with before and after processing superimposed
    for i in range(n_channels):
        # Get axis for current channel
        ax = axes[i]

        # Plot raw data
        ax.plot(time, np.abs(emg_data[:, i]), "b-", alpha=0.7, label="Raw")

        # Plot processed data
        ax.plot(
            time[:: len(emg_data) // len(processed_data)],
            processed_data[:, i],
            "r-",
            alpha=0.7,
            label="Processed",
        )

        # plot a vertical line every 5 seconds
        for j in range(0, len(time), 5 * 1000):
            ax.axvline(x=time[j], color="gray", linestyle="--", alpha=0.5)

        # Set title and labels
        ax.set_title(f"Channel {i+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_channels, rows * cols):
        axes[i].set_visible(False)

    plt.tight_layout()
    # plt.savefig("emg_channel_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
