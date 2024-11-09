import os
import subprocess as sp
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import emager_py.transforms as et
import emager_py.dataset as ed
import emager_py.data_processing as dp
from emager_py.streamers import RedisStreamer
import emager_py.majority_vote as mv
import emager_py.finn.remote_operations as ro

import globals
import utils


def get_and_save_best_model(subject: int, quant: int, shots: int, transform_name: str):
    """
    Find the best cross-validation model for the given parameters and save it to:
    `f"{globals.OUT_DIR_ROOT}/{globals.OUT_DIR_FINN}/finn_config.json"`
    """
    best_ses, best_rep, _ = utils.get_best_model(
        subject,
        quant,
        shots,
        utils.ModelMetric.ACC_MAJ_INTER,
    )

    params_dict = {
        "subject": subject,
        "quantization": quant,
        "shots": shots,
        "session": best_ses,
        "repetition": best_rep,
        "transform": transform_name,
    }
    
    os.makedirs(globals.OUT_DIR_ROOT + globals.OUT_DIR_FINN, exist_ok=True)
    
    with open(
        globals.OUT_DIR_ROOT + globals.OUT_DIR_FINN + "finn_config.json", "w"
    ) as f:
        json.dump(params_dict, f)

    return params_dict


def launch_finn_build():
    """
    Build the FINN accelerator.

    First, gets the best model from all cross-validations (across session+repetitions) and writes its info on disk.

    Then, executes `src/build_dataflow.py` with the FINN command-line entry point, which reads said best model file.
    """
    proj_dir = os.getcwd()

    # must chdir, since finn's script uses relative paths...
    os.chdir(globals.FINN_ROOT)
    cmd = [
        "bash",
        "-c",
        # Can't separate the arguments since we call bash -c
        f"./run-docker.sh build_custom {proj_dir} src/build_dataflow",
    ]
    try:
        ret = sp.run(cmd, universal_newlines=True)
        if ret.returncode != 0:
            raise RuntimeError("Failed to build FINN accelerator")
    except:  # noqa
        os.chdir(proj_dir)


def test_finn_accelerator(hostname: str, simulate_results: bool = False):
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
    rs.clear()
    rs.r.set_pynq_params(md["transform"])
    calib_data, test_data = ed.get_lnocv_datasets(
        globals.EMAGER_DATASET_ROOT, md["subject"], test_session, md["repetition"]
    )
    calib_data, calib_labels = dp.extract_labels(calib_data)
    test_data, test_labels = dp.extract_labels(test_data)
    decim = et.get_transform_decimation(md["transform"])
    test_labels = test_labels[::decim]

    # Take some calibration data
    for i in np.unique(calib_labels):
        matches = np.argwhere(calib_labels == i).flatten()
        idxs = np.random.choice(
            matches, md["shots"] * globals.EMAGER_SAMPLE_BATCH, replace=False
        )
        for idx in idxs:
            rs.r.push_sample(calib_data[idx], calib_labels[idx])

    # Push test data
    list(map(lambda x: rs.r.push_sample(x), test_data))

    # Get predictions
    if simulate_results:
        inferences_to_push = len(test_labels)
        for _ in range(inferences_to_push):
            rs.r.r.lpush(rs.r.PREDICTIONS_FIFO_KEY, np.random.randint(0, 6))
    else:
        c = ro.connect_to_pynq()
        ret = ro.run_remote_finn(
            c, globals.TARGET_EMAGER_PYNQ_PATH, "python3 validate_finn.py"
        )

        # ret = ro.run_remote_finn(c, globals.TARGET_EMAGER_PYNQ_PATH, "ls -la")
        c.close()

    # Retrieve labels from Redis
    pred_labels = np.array(
        [
            rs.r.decode_label_bytes(p)
            for p in rs.r.r.rpop(
                rs.r.PREDICTIONS_FIFO_KEY, rs.r.r.llen((rs.r.PREDICTIONS_FIFO_KEY))
            )
        ]
    ).squeeze()

    # Calculate the performance
    n_votes = globals.EMAGER_SAMPLING_RATE * 150 // (decim * 1000)
    true_maj = mv.majority_vote(test_labels, n_votes)
    pred_maj = mv.majority_vote(pred_labels, n_votes)

    acc_raw = accuracy_score(test_labels, pred_labels, normalize=True)
    acc_maj = accuracy_score(true_maj, pred_maj, normalize=True)

    ret = {
        "shots": [md["shots"]],
        utils.ModelMetric.ACC_RAW_INTER: [acc_raw],
        utils.ModelMetric.ACC_MAJ_INTER: [acc_maj],
    }

    print(ret)

    # Write results to disk
    out_path = utils.format_model_name(
        globals.OUT_DIR_FINN,
        md["subject"],
        md["session"],
        md["repetition"],
        md["quantization"],
        md["shots"],
    )
    results_df = pd.DataFrame(ret)
    results_df.to_csv(out_path, index=False)
    return results_df


if __name__ == "__main__":
    from emager_py.emager_redis import get_docker_redis_ip
    from multiprocessing import Process
    import time

    quants = [2, 3, 4, 6, 8]
    subjects = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # subjects = [9]
    # quants = [4]

    for sub in subjects:
        procs = []
        for quant in quants:
            get_and_save_best_model(sub, quant, globals.SHOTS, globals.TRANSFORM)
            utils.lock_finn(subject=sub, quant=quant, shots=globals.SHOTS)

            p = Process(target=launch_finn_build, args=())
            p.start()
            procs.append(p)

            while utils.is_finn_locked():
                time.sleep(1)  # wait for FINN to start and read finn_config.json

        for p in procs:
            p.join()
            
    # test_finn_accelerator(get_docker_redis_ip(), True)
    # test_finn_accelerator("pynq", False)

    print("Exiting.")
