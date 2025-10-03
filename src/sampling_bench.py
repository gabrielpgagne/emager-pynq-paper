import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


if __name__ == "__main__":
    results = pd.read_csv("data/sampling_benchmark.csv")
    results = results.groupby("batch").mean()

    print(results)

    max_batch = results.index.max()
    max_sps = results["sps"].max()

    # font = {"size": 24}
    # matplotlib.rc("font", **font)

    plt.figure(figsize=(16, 9))
    sns.set_theme(font_scale=2)
    sns.set_style("whitegrid")

    ax = sns.lineplot(results[1:], x="batch", y="sps", marker="o", linewidth=2)
    ax.spines[["right", "top"]].set_visible(False)

    l1 = ax.axhline(
        1000,
        0,
        max_batch,
        color="r",
        linestyle="dashed",
    )
    l2 = ax.axhline(
        max_sps,
        0,
        max_batch,
        color="b",
        linestyle="dashed",
    )

    ax.legend(
        [l1, l2],
        ["Minimum required (1000 sps)", f"Maximum measured ({max_sps:.0f} sps)"],
    )
    ax.set_yticks(range(0, 11000, 1000))
    # ax.set_xticks(range(0, max_batch + 1, 25))

    ax.grid(True, "both", "both")
    ax.set_ylabel("Samples per second")
    ax.set_xlabel("Batch size")
    # plt.savefig("output/img/sampling_throughput.png", bbox_inches="tight")
    plt.show()
