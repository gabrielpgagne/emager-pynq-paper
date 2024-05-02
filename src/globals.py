import emager_py.utils
import sys

if sys.platform == "darwin":
    emager_py.utils.DATASETS_ROOT = "/Users/gabrielgagne/Documents/Datasets/"

# emager_py.utils.set_logging()

EMAGER_DATASET_ROOT = emager_py.utils.DATASETS_ROOT + "EMAGER/"

OUT_DIR_MODELS = "./output/models/"
OUT_DIR_STATS = "./output/eval/"
