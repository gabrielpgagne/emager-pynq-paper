import os
import subprocess as sp

import globals

if __name__ == "__main__":
    # TODO probably eval almost everything in build_dataflow.py
    # TODO some sort of hard-coded config file
    # FIXME copy the EMaGer dataset subject somewhere in this project
    
    cmd = [
        "bash",
        globals.FINN_ROOT + "run-docker.sh",
        "build_custom",
        os.getcwd(),
        "src/build_dataflow",
    ]
    print(" ".join(cmd))
    ret = sp.run(cmd, check=True, universal_newlines=True)
    if ret.returncode != 0:
        raise RuntimeError("Failed to build FINN accelerator")
