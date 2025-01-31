import emager_py.transforms
import emager_py.utils
import sys


# emager_py.utils.set_logging()

SUBJECT = 0
QUANT = 8
SHOTS = -1

EMAGER_DATA_SHAPE = (4, 16)
EMAGER_SAMPLING_RATE = 1000
EMAGER_SAMPLE_BATCH = 25
TRANSFORM = "root"

assert (
    TRANSFORM in emager_py.transforms.transforms_lut
), f"Invalid transform: {TRANSFORM}. Must be in {emager_py.transforms.transforms_lut.keys()}"


EMAGER_DATASET_ROOT = "./data/EMAGER/"
if sys.platform == "darwin":
    "/Users/gabrielgagne/Documents/Datasets/EMAGER/"

VALIDATION_EMAGER_ROOT = "./data/EMAGER/"

# PYNQ_HOSTNAME = "pynq.local"
PYNQ_HOSTNAME = "192.168.0.99"

FINN_ROOT = "/home/gabrielgagne/Documents/git/finn/"
FINN_TARGET_BOARD = "zybo-z7-20"
FINN_MODEL_PARAMS_DICT = {
    "subject": -1,
    "quantization": -1,
    "shots": -1,
}

TARGET_EMAGER_PYNQ_PATH = "/home/xilinx/workspace/emager-pynq/"

# OUT_DIR_ROOT = "./output_scnn_16_16_16_32_32_q/"
# OUT_DIR_ROOT = "./output_inter_subject/"
# OUT_DIR_ROOT = "./output_scnn_16_16_16_32_32/"
# OUT_DIR_ROOT = "./output_scnn/"
# OUT_DIR_ROOT = "./output_cnn/"
OUT_DIR_ROOT = "./output/"

OUT_DIR_MODELS = "models/"
OUT_DIR_STATS = "eval/"
OUT_DIR_FINN = "finn/"
