import numpy as np
import matplotlib.pyplot as plt


from emager_py.finn import remote_operations as ro
import emager_py.screen_guided_training as sgt
import emager_py.emager_redis as er
import emager_py.dataset as ed

import globals as g


def sample_sgt():
    SUBJECT = 13
    SESSION = 2
    N_REPS = 5
    REP_TIME = 5

    hostname = g.PYNQ_HOSTNAME

    images_path = "output/gestures/"
    gestures = [2, 14, 26, 1, 8, 30]

    finetune_data_dir = "data/EMAGER/"

    r = er.EmagerRedis(hostname)
    r.set_pynq_params(g.TRANSFORM)
    r.set_sampling_params(1000, 25, 5000)
    r.set_rhd_sampler_params(
        low_bw=15,
        hi_bw=350,
        # en_dsp=1,
        fp_dsp=20,
        bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
    )
    r.clear_data()

    c = ro.connect_to_pynq(hostname=hostname)
    ro.sample_training_data(r, c, 1000, g.TARGET_EMAGER_PYNQ_PATH, 0)  # warmup
    r.clear_data()

    global data
    data = np.zeros((len(gestures), 0, REP_TIME * 1000, 64))

    def resume_training_cb(gesture_id):
        global data

        ro.sample_training_data(
            r,
            c,
            REP_TIME * g.EMAGER_SAMPLING_RATE,
            g.TARGET_EMAGER_PYNQ_PATH,
            gesture_id,
        )
        if gesture_id == len(gestures) - 1:
            new_data = r.dump_labelled_to_numpy(False)
            data = np.concatenate((data, new_data), axis=1)
            noise_floor = np.sqrt(np.mean((data[4] - np.mean(data[4])) ** 2))
            print(f"RMS Noise floor: {noise_floor:.2f}")
            # On MacOS, plotting crashes...
            # plt.figure()
            # plot_data = new_data.reshape((-1, 64))
            # for i in range(4):
            #     for j in range(16):
            #         plt.subplot(4, 16, i * 16 + j + 1)
            #         plt.plot(plot_data[:, i * 16 + j])
            # plt.show()

    # ========== TRAINING ==========

    sgt.EmagerGuidedTraining(
        N_REPS,
        gestures,
        images_path,
        5,
        resume_training_callback=resume_training_cb,
        callback_arg="gesture",
    ).start()

    # Save unprocessed data
    ed.process_save_dataset(data, finetune_data_dir, lambda d: d, SUBJECT, SESSION)


if __name__ == "__main__":
    # utils.set_logging()

    # hostname = er.get_docker_redis_ip()
    sample_sgt()
