import lightning as L
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

from emager_py.finn import remote_operations as ro
import emager_py.screen_guided_training as sgt
import emager_py.emager_redis as er
import emager_py.data_processing as dp
import emager_py.dataset as ed
import emager_py.transforms as etrans
import emager_py.torch.models as etm
from emager_py import utils

import globals as g
from finn_build import launch_finn_build

if __name__ == "__main__":
    utils.set_logging()

    # hostname = er.get_docker_redis_ip()
    hostname = "pynq"

    subject = 13
    session = 1
    quant = 8
    shots = 10
    transform = etrans.transforms_lut[g.TRANSFORM]
    finetune_data_dir = "data/finetune/"
    n_reps = 1
    rep_time = 3

    r = er.EmagerRedis(hostname)
    r.set_pynq_params(g.TRANSFORM)
    r.set_rhd_sampler_params(
        bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH
        + "bitfile/finn-accel.bit"
        # bitstream=""
    )

    c = ro.connect_to_pynq()

    def resume_training_cb(gesture_id):
        ro.sample_training_data(
            c,
            hostname,
            rep_time * g.EMAGER_SAMPLING_RATE,
            g.TARGET_EMAGER_PYNQ_PATH,
            gesture_id,
        )

    # ========== TRAINING ==========

    # r.clear_data()

    # sgt.EmagerGuidedTraining(
    #     n_reps,
    #     gestures_path="output/gestures/",
    #     resume_training_callback=resume_training_cb,
    #     callback_arg="gesture",
    # ).start()

    # # raw data -> processed DataLoader
    # data = r.dump_labelled_to_numpy()
    # print(data.shape)

    # data_dir = ed.process_save_dataset(
    #     data, finetune_data_dir, transform, subject, session
    # )
    # data = ed.load_emager_data(data_dir, subject, session)

    # data, labels = dp.extract_labels_and_roll(data, 2)
    # data = data.astype(np.float32)
    # train_triplets = dp.generate_triplets(data, labels, 5000)
    # train_triplets = [
    #     torch.from_numpy(t).reshape((-1, 1, *g.EMAGER_DATA_SHAPE))
    #     for t in train_triplets
    # ]
    # train_dl = DataLoader(TensorDataset(*train_triplets), batch_size=32, shuffle=True)

    # # Train
    # trainer = L.Trainer(
    #     accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
    #     enable_checkpointing=False,
    #     logger=True,
    #     max_epochs=10,
    # )
    # model = etm.EmagerSCNN(quant)
    # trainer.fit(model, train_dl)
    # launch_finn_build(subject, quant, shots, g.TRANSFORM)

    # =========== TESTING ==============

    # Get some calibration data
    USE_LIVE_DATA = False

    if USE_LIVE_DATA:
        # TODO doesnt seem to work
        sgt.EmagerGuidedTraining(
            1,
            gestures_path="output/gestures/",
            resume_training_callback=resume_training_cb,
            callback_arg="gesture",
        ).start()
    else:
        # works!
        calib_data, test_data = ed.get_lnocv_datasets(
            g.EMAGER_DATASET_ROOT, 0, session, [0, 1]
        )
        calib_data, calib_labels = dp.extract_labels(calib_data)
        for i in np.unique(calib_labels):
            matches = np.argwhere(calib_labels == i).flatten()
            idxs = np.random.choice(
                matches, shots * g.EMAGER_SAMPLE_BATCH, replace=False
            )
            for idx in idxs:
                r.push_sample(calib_data[idx], calib_labels[idx])

    # After training just remotely run the live test client
    r.set_sampling_params(n_samples=-1)
    ro.run_remote_finn(
        ro.connect_to_pynq(), g.TARGET_EMAGER_PYNQ_PATH, "python3 client.py"
    )

    # TODO script to re-calibrate with embeddings, maybe button input on PYNQ board?
