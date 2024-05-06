import emager_py.transforms
import emager_py.utils
import sys

if sys.platform == "darwin":
    emager_py.utils.DATASETS_ROOT = "/Users/gabrielgagne/Documents/Datasets/"

# emager_py.utils.set_logging()

SUBJECT = 0
QUANT = 4
SHOTS = 10

EMAGER_DATA_SHAPE = (4, 16)
TRANSFORM = "root"

assert (
    TRANSFORM in emager_py.transforms.transforms_lut
), f"Invalid transform: {TRANSFORM}. Must be in {emager_py.transforms.transforms_lut.keys()}"


EMAGER_DATASET_ROOT = emager_py.utils.DATASETS_ROOT + "EMAGER/"
VALIDATION_EMAGER_ROOT = "./data/EMAGER/"

FINN_ROOT = "/home/gabrielgagne/Documents/git/finn/"
FINN_TARGET_BOARD = "zybo-z7-20"
FINN_MODEL_PARAMS_DICT = {
    "subject": -1,
    "quantization": -1,
    "shots": -1,
}

TARGET_EMAGER_PYNQ_PATH = "/home/xilinx/workspace/emager-pynq/"

OUT_DIR_ROOT = "./output/"
OUT_DIR_MODELS = "models/"
OUT_DIR_STATS = "eval/"
OUT_DIR_FINN = "finn/"
