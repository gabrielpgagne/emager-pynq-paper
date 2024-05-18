import os
import subprocess as sp
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import emager_py.transforms as et
import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.data_generator as dg
from emager_py.streamers import RedisStreamer
import emager_py.majority_vote as mv
import emager_py.finn.remote_operations as ro

import globals
import utils


def launch_finn_build(subject: int, quant: int, shots: int, transform_name: str):
    """
    Build the FINN accelerator.

    First, gets the best model from all cross-validations (across session+repetitions) and writes its info on disk.

    Then, executes `src/build_dataflow.py` with the FINN command-line entry point, which reads said best model file.
    """
    best_ses, best_rep, _ = utils.get_best_model(
        subject,
        quant,
        shots,
        utils.ModelMetric.ACC_MAJ,
    )

    params_dict = {
        "subject": subject,
        "quantization" : quant,
        "shots" : shots,
        "session": best_ses,
        "repetition": best_rep,
        "transform": transform_name
    }
    with open(
        globals.OUT_DIR_ROOT + globals.OUT_DIR_FINN + "finn_config.json", "w"
    ) as f:
        json.dump(params_dict, f)

    cmd = [
        "bash",
        "-c",
        f"{globals.FINN_ROOT}/run-docker.sh build_custom {os.getcwd()} src/build_dataflow",
    ]

    ret = sp.run(cmd, check=True, universal_newlines=True)
    if ret.returncode != 0:
        raise RuntimeError("Failed to build FINN accelerator")


def test_finn_accelerator(hostname:str, simulate_results: bool = False):
    """
    Assumes the output/finn/finn_config.json dict is the most up-to-date model on remote device.

    Test the FINN accelerator:
        - Load the inter-session EMaGer data
        - Connect via Redis
        - Push the setup keys and data
        - Run the inference script remotely
        - Fetch back the results
        - Write accuracy to the FINN output folder
    
    Params:
        - hostname: redis hostname to connect to
    """
    md = utils.get_model_params_from_disk()

    # Load test data
    test_session = 2 if md["session"] == 1 else 1

    # Push data as fast as possible
    rs = RedisStreamer(hostname, False)
    rs.r.set_pynq_params(md["transform"])
    datagen = dg.EmagerDataGenerator(rs, globals.EMAGER_DATASET_ROOT, 1000000, globals.EMAGER_SAMPLE_BATCH, False, False)
    datagen.prepare_data(md["subject"], test_session)
    datagen.serve_data()
    
    # Get inferences
    true_labels = datagen.labels[:, 0, :]
    if simulate_results:
        inferences_to_push = len(true_labels)
        for _ in range(inferences_to_push):
            rs.r.r.lpush(rs.r.PREDICTIONS_FIFO_KEY, np.random.randint(0, 6))
    else:
        c = ro.connect_to_pynq()
        ret = ro.run_remote_finn(
            c, globals.TARGET_EMAGER_PYNQ_PATH, "python3 validate_finn.py"
        )
        c.close()

    # Retrieve labels from Redis
    pred_labels = np.array([int(p) for p in rs.r.r.rpop(rs.r.PREDICTIONS_FIFO_KEY, rs.r.r.llen((rs.r.PREDICTIONS_FIFO_KEY)))])

    # Calculate the performance
    n_votes = globals.EMAGER_SAMPLING_RATE * 150 // (et.get_transform_decimation(md["transform"]) * 1000)
    true_maj = mv.majority_vote(true_labels, n_votes)
    pred_maj =  mv.majority_vote(pred_labels, n_votes)

    acc_raw = accuracy_score(true_labels, pred_labels, normalize=True)
    acc_maj = accuracy_score(true_maj, pred_maj, normalize=True)

    ret = {
        "shots": [md["shots"]],
        utils.ModelMetric.ACC_RAW: [acc_raw],
        utils.ModelMetric.ACC_MAJ: [acc_maj],
    }

    # Write results to disk
    out_dir = utils.format_finn_output_dir(md["subject"], md["quantization"], md["shots"])
    file_name = utils.format_model_name(md["subject"], md["session"], md["repetition"], md["quantization"], ".csv", True).split("/")[-1]
    results_df = pd.DataFrame(ret)
    results_df.to_csv(out_dir+file_name, index=False)
    return results_df


if __name__ == "__main__":
    from emager_py.emager_redis import get_docker_redis_ip
    # TODO: after building, test the accelerator
    launch_finn_build(globals.SUBJECT, globals.QUANT, globals.SHOTS, globals.TRANSFORM)
    print(test_finn_accelerator(get_docker_redis_ip(), True))
    print("Exiting.")
