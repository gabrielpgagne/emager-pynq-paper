import matplotlib.pyplot as plt
import seaborn as sns

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

sns.set_theme(font_scale=2)
sns.lineplot(results[1:], x="Batchsize", y="sps", linewidth=2)
plt.axhline(
    1000,
    0,
    max_batch,
    color="r",
    linestyle="dashed",
)
plt.axhline(
    max_sps,
    0,
    max_batch,
    color="b",
    linestyle="dashed",
)

plt.ylim(0, 8300)
plt.grid(True, "both", "both")
plt.ylabel("Samples per second")
plt.xlabel("Batch size")
plt.savefig("output/img/sampling_throughput.png",bbox_inches="tight")

plt.show()
