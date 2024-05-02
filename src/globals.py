import emager_py.utils
import sys

if sys.platform == "darwin":
    emager_py.utils.DATASETS_ROOT = "/Users/gabrielgagne/Documents/Datasets/"

# emager_py.utils.set_logging()

EMAGER_DATA_SHAPE = (4, 16)
TRANSFORM = "default"

TARGET_BOARD = "zybo-z7-20"

EMAGER_DATASET_ROOT = emager_py.utils.DATASETS_ROOT + "EMAGER/"
VALIDATION_EMAGER_ROOT = "./data/EMAGER/"

FINN_ROOT = "/home/gabrielgagne/Documents/git/finn/"

TARGET_EMAGER_PYNQ_PATH = "/home/xilinx/workspace/emager-pynq/"

OUT_DIR_MODELS = "./output/models/"
OUT_DIR_STATS = "./output/eval/"
OUT_DIR_FINN = "./output/finn/"
