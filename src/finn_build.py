import os
import subprocess as sp
import json

import emager_py.utils as eu
import emager_py.finn.remote_operations as ro

import globals
import utils


def launch_finn_build(subject, quant, shots):
    globals.FINN_MODEL_PARAMS_DICT["subject"] = subject
    globals.FINN_MODEL_PARAMS_DICT["quantization"] = quant
    globals.FINN_MODEL_PARAMS_DICT["shots"] = shots

    with open(
        globals.OUT_DIR_ROOT + globals.OUT_DIR_FINN + "finn_config.json", "w"
    ) as f:
        json.dump(globals.FINN_MODEL_PARAMS_DICT, f)

    cmd = [
        "bash",
        "-c",
        f"{globals.FINN_ROOT}/run-docker.sh build_custom {os.getcwd()} src/build_dataflow",
    ]

    ret = sp.run(cmd, check=True, universal_newlines=True)
    if ret.returncode != 0:
        raise RuntimeError("Failed to build FINN accelerator")


def test_finn_accelerator():
    model_params_dict = dict()
    with open(
        globals.OUT_DIR_ROOT + globals.OUT_DIR_FINN + "finn_config.json", "r"
    ) as f:
        model_params_dict = json.load(f)
    valid_data_dir = utils.format_finn_output_dir(
        model_params_dict["subject"],
        model_params_dict["quantization"],
        model_params_dict["shots"],
    )
    valid_data_files = [
        "finn_preproc_valid_data.npz",
        "finn_raw_valid_data.npz",
        "finn_transform_fn_name.txt",
    ]
    c = ro.connect_to_pynq()
    for f in valid_data_files:
        c.put(valid_data_dir + f, remote=globals.TARGET_EMAGER_PYNQ_PATH)
    ret = ro.run_remote_finn(
        c, globals.TARGET_EMAGER_PYNQ_PATH, "python3 validate_finn.py"
    )
    c.close()

    with open(valid_data_dir + "finn_validation_results.txt", "w") as f:
        f.write(ret.stdout)


if __name__ == "__main__":
    # TODO: after building, test the accelerator
    launch_finn_build(globals.SUBJECT, globals.QUANT, globals.SHOTS)
    test_finn_accelerator()
    print("Exiting.")
