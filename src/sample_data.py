import os

from emager_py.finn import remote_operations as ro
import emager_py.screen_guided_training as sgt
import emager_py.emager_redis as er
import emager_py.dataset as ed

import globals as g


def sample_sgt(
    r: er.EmagerRedis,
    hostname,
    data_dir,
    subject,
    session,
    n_reps,
    rep_time,
    gestures=g.GESTURES,
    gestures_dir=g.GESTURES_PATH,
    use_dsp=False,
):
    r.set_sampling_params(1000, 25, 5000)
    r.set_rhd_sampler_params(
        low_bw=15,
        hi_bw=350,
        en_dsp=int(use_dsp),
        fp_dsp=20,
        bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
    )

    c = ro.connect_to_pynq(hostname=hostname)
    ro.sample_training_data(r, c, 1000, g.TARGET_EMAGER_PYNQ_PATH, 0)  # warmup
    r.clear_data()

    def resume_training_cb(gesture_id):
        ro.sample_training_data(
            r,
            c,
            rep_time * g.EMAGER_SAMPLING_RATE,
            g.TARGET_EMAGER_PYNQ_PATH,
            gesture_id,
        )

    # ========== TRAINING ==========

    sgt.EmagerGuidedTraining(
        n_reps,
        gestures,
        gestures_dir,
        rep_time,
        resume_training_callback=resume_training_cb,
        callback_arg="gesture",
    ).start()

    # Save unprocessed data
    data = r.dump_labelled_to_numpy(False)
    data[..., 14] = 0  # bad channel

    data_path = data_dir + ed.format_subject(subject) + ed.format_session(session)
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            print(f"Removing {f}")
            os.remove(os.path.join(data_path, f))

    ed.process_save_dataset(data, data_dir, lambda d: d, subject, session)
    return data


if __name__ == "__main__":
    SUBJECT = 14
    SESSION = 1

    N_REPS = 1
    REP_TIME = 5

    DATA_DIR = "data/live_test/"

    HOSTNAME = g.PYNQ_HOSTNAME
    GESTURES_ID = g.GESTURES
    IMAGES_DIR = g.GESTURES_PATH

    r = er.EmagerRedis(HOSTNAME)

    sample_sgt(
        r,
        HOSTNAME,
        DATA_DIR,
        SUBJECT,
        SESSION,
        N_REPS,
        REP_TIME,
        g.GESTURES,
        g.GESTURES_PATH,
        False,
    )
