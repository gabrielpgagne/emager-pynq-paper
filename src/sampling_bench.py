import matplotlib.pyplot as plt

import pandas as pd

results = dict()
with open("data/sampling_throughput.txt") as f:
    lines = f.readlines()
    for line in lines:
        toks = line.split()
        results[toks[0]] = [int(tok) for tok in toks[1:]]

max_sps = results["sps"][0]
max_batch = results["Batchsize"][-1]

results = pd.DataFrame(results)

results[1:].plot(x="Batchsize", y=["sps"])

plt.ylabel("Samples per second")
plt.hlines(
    1000,
    0,
    max_batch,
    colors="r",
    linestyles="dashed",
)
plt.hlines(
    max_sps,
    0,
    max_batch,
    colors="b",
    linestyles="dashed",
)
plt.grid(True, "both", "both")
plt.legend(
    [
        "Throughput (samples per second)",
        "Minimum target sampling rate",
        "Sampling rate without Redis",
    ],
    loc="center right",
)
plt.savefig("output/img/sampling_throughput.png")
plt.show()
