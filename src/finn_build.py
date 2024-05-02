import os
import subprocess as sp

import emager_py.utils as eu
import emager_py.dataset as ed

import utils
import globals

if __name__ == "__main__":
    # TODO probably eval almost everything in build_dataflow.py
    # TODO some sort of hard-coded config file

    eu.set_logging()

    SUBJECT = 0
    QUANT = 4
    SHOTS = 10

    session, lnocv, _ = utils.get_best_model(
        SUBJECT, QUANT, SHOTS, utils.ModelMetric.ACC_MAJ
    )
    """
    ed.load_process_save_dataset(
        globals.EMAGER_DATASET_ROOT,
        globals.VALIDATION_EMAGER_ROOT,
        "default",
        SUBJECT,
        session,
    )
    """
    ed.load_emager_data(globals.VALIDATION_EMAGER_ROOT, SUBJECT, session)

    cmd = [
        "bash",
        globals.FINN_ROOT + "run-docker.sh",
        "build_custom",
        os.getcwd(),
        "src/build_dataflow",
    ]
    # print(" ".join(cmd))
    ret = sp.run(cmd, check=True, universal_newlines=True)
    if ret.returncode != 0:
        raise RuntimeError("Failed to build FINN accelerator")
