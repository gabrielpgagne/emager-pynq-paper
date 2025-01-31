import numpy as np
import matplotlib.pyplot as plt
import serial

import emager_py.screen_guided_training as sgt
import emager_py.dataset as ed
from emager_py.utils import EMAGER_CHANNEL_MAP


def decode_buffer(data: np.ndarray):
    """
    data is a buffer of uint8 with size (n*128)

    Ch0 LSb is set to 0, the rest are set to 1

    Returns the decoded EMG data with shape (n, 64)
    """
    n = int(len(data) // 128)
    idx = np.argwhere(data & 1 == 0)

    # Check even indices first
    even_idx = idx[idx % 2 == 0]
    roll = even_idx.item(0) if len(even_idx) == n else -1

    # Only check odd indices if even indices failed
    if roll == -1:
        odd_idx = idx[idx % 2 == 1]
        roll = odd_idx.item(0) if len(odd_idx) == n else -1

    if roll == -1:
        return np.zeros((0, 64), dtype=np.int16)

    rolled = np.roll(data, -roll * n)
    return np.frombuffer(rolled, dtype="<i2").reshape(-1, 64)


def sample_sgt():
    SUBJECT = 14
    SESSION = 2
    N_REPS = 1
    REP_TIME = 2

    images_path = "output/gestures/"
    gestures = [2, 14, 26, 1, 8, 30]

    finetune_data_dir = "data/EMAGER/"

    global data, labels
    data = []

    ser = serial.Serial("/dev/cu.usbmodem1403", 1500000)
    ser.close()

    def resume_training_cb(gesture_id):
        global data

        ser.open()
        while len(data) < (gesture_id + 1) * REP_TIME * 1000:
            d_ret = np.frombuffer(ser.read(128), dtype=np.uint8)
            pkt = decode_buffer(d_ret)
            if len(pkt) == 0:
                continue
            data.append(pkt)
        ser.close()
        noise_floor = np.sqrt(
            np.mean((data[gesture_id] - np.mean(data[gesture_id])) ** 2)
        )
        print(f"RMS Noise floor: {noise_floor:.2f}")

        print(f"Done {gesture_id}")

    # ========== TRAINING ==========

    sgt.EmagerGuidedTraining(
        N_REPS,
        gestures,
        images_path,
        REP_TIME,
        resume_training_callback=resume_training_cb,
        callback_arg="gesture",
    ).start()

    data = np.array(data).reshape((len(gestures), N_REPS, -1, 64))
    data = data[..., EMAGER_CHANNEL_MAP]

    # Save unprocessed data
    ed.process_save_dataset(data, finetune_data_dir, lambda d: d, SUBJECT, SESSION)


if __name__ == "__main__":
    # utils.set_logging()

    # hostname = er.get_docker_redis_ip()
    sample_sgt()
