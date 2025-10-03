import numpy as np
import matplotlib.pyplot as plt

import emager_py.dataset as ed
import emager_py.data_processing as dp

from emager_py.transforms import default_processing
from emager_py.quantization import nroot_c


def plot_root_i16_vs_u8(bits):
    """Plot the difference between int16 and uint8 data.

    Returns:
        tuple: Figure and axis objects
    """
    # roots = np.arange(1.5, 2.5, 0.1)
    roots = [1.5, 2.0, 2.5, 3.0]
    x = np.arange(0, 32767, 1)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hlines(
        1 << (bits - 1),
        0,
        np.amax(x),
        color="red",
        linestyle="--",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for e in roots:
        y = np.pow(x, 1 / e)
        ax.plot(x, y, label=f"r = {e:.1f}")

    ax.set_xlabel("Valeur d'origine absolue (i16)")
    ax.set_ylabel(f"Valeur quantifiée absolue (i{bits})")
    ax.set_ylim(0, 1.1 * (1 << (bits - 1)))
    ax.legend()
    ax.grid(True)

    return fig, ax


def plot_data_histogram(data, root=1.7, bits=8, bins=50):
    """Plot histograms of data before processing, after processing, and after quantization.

    Args:
        root (float): Root value for quantization (default: 1.7)
        bits (int): Number of bits for quantization (default: 8)
        bins (int): Number of histogram bins (default: 50)

    Returns:
        tuple: Figure and axes objects
    """

    # Process data
    processed_data = default_processing(data)

    # Quantize data
    quantized_data = nroot_c(processed_data, root, bits)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Flatten data for histogram plotting
    raw_flat = data.flatten()
    processed_flat = processed_data.flatten()
    quantized_flat = quantized_data.flatten()

    # Plot histogram of raw data
    axes[0].hist(raw_flat, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_title("Données brutes")
    axes[0].set_xlabel("Valeur")
    axes[0].set_ylabel("Fréquence")
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Add statistics
    axes[0].text(
        0.02,
        0.98,
        f"Mean: {raw_flat.mean():.2f}\nStd: {raw_flat.std():.2f}\nMin: {raw_flat.min():.2f}\nMax: {raw_flat.max():.2f}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot histogram of processed data
    axes[1].hist(processed_flat, bins=bins, alpha=0.7, color="green", edgecolor="black")
    axes[1].set_title("Données traitées")
    axes[1].set_xlabel("Valeur")
    axes[1].set_ylabel("Fréquence")
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Add statistics
    axes[1].text(
        0.02,
        0.98,
        f"Mean: {processed_flat.mean():.2f}\nStd: {processed_flat.std():.2f}\nMin: {processed_flat.min():.2f}\nMax: {processed_flat.max():.2f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Plot histogram of quantized data
    axes[2].hist(quantized_flat, bins=bins, alpha=0.7, color="red", edgecolor="black")
    axes[2].set_title(f"Données quantifiées (r={root}, {bits}-bit)")
    axes[2].set_xlabel("Valeur")
    axes[2].set_ylabel("Fréquence")
    axes[2].grid(True, alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    # Add statistics
    axes[2].text(
        0.02,
        0.98,
        f"Mean: {quantized_flat.mean():.2f}\nStd: {quantized_flat.std():.2f}\nMin: {quantized_flat.min():.2f}\nMax: {quantized_flat.max():.2f}",
        transform=axes[2].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Overall title
    fig.suptitle(
        "Histogram of EMG data",
        fontsize=16,
    )

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    # Example usage of the histogram function
    # Uncomment the lines below to generate histogram plots for a specific subject and session
    # fig, axes = plot_data_histogram(subject=0, session="001", root=1.7, bits=8, bins=50)
    # plt.show()

    data = [
        ed.load_emager_data("data/EMAGER/", sub, ses)
        for sub in range(13)
        for ses in ["001", "002"]
    ]
    data = np.array(data).reshape(-1, 64)

    # data = dp.filter_data(data)
    # data = np.abs(data)

    data = default_processing(data)

    print(f"Loaded data for histogram plotting with shape {data.shape}")

    print(
        f"Dataset stats: Mean: {data.mean():.2f}, Std: {data.std():.2f}, Max: {data.max():.2f}, Min: {data.min():.2f}"
    )

    # plot_data_histogram(data, root=1.7, bits=8, bins=256)
    # plt.show()

    print(
        f"raw data: mean {data.mean():.2f} +/- {data.std():.2f}, 95th {np.percentile(data, 95):.2f}, max {data.max():.2f}"
    )

    for n in np.arange(1.1, 2.1, 0.1):
        rdata = nroot_c(data, n, bits=8)
        print(
            f"{n:.1f}th-root data: mean {rdata.mean():.2f} +/- {rdata.std():.2f}, 95th {np.percentile(data, 95):.2f}, max {data.max():.2f}"
        )
