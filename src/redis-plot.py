from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import emager_py.finn.remote_operations as ro
import threading
import time

from emager_py.streamers import RedisStreamer
import globals as g

if __name__ == "__main__":
    FS = 1000
    BATCH = 50

    HOST = g.PYNQ_HOSTNAME

    c = ro.connect_to_pynq(hostname=HOST)
    t = threading.Thread(
        target=ro.run_remote_finn,
        args=(c, ro.DEFAULT_EMAGER_PYNQ_PATH, "rhd_sampler"),
        daemon=True,
    ).start()

    time.sleep(1)

    print("Starting client and oscilloscope...")
    stream_client = RedisStreamer(HOST, False)
    stream_client.r.set_rhd_sampler_params(
        15, 450, 0, 20, "/home/xilinx/workspace/emager-pynq/bitfile/finn-accel.bit"
    )
    stream_client.r.set_sampling_params(1000, BATCH, 120 * FS)
    stream_client.r.clear_data()

    data = []
    t0 = time.perf_counter()
    while len(data) < 5000:
        new_data = stream_client.read()
        if len(new_data) == 0:
            continue
        elif len(data) == 0:
            t0 = time.perf_counter()
            print(len(new_data))
        data.extend(new_data)
        # print(len(data))
    t1 = time.perf_counter()
    print(f"Elapsed time: {t1 - t0:.3f} s")
    print(f"Actual sample rate: {len(data) / (t1 - t0):.3f} Hz")
    data = np.array(data)
    # filt = signal.iirnotch(60, 5, FS)
    # data = signal.filtfilt(filt[0], filt[1], data, axis=0)

    plt.figure(1)
    for i in range(64):
        plt.subplot(4, 16, i + 1)
        plt.plot(np.linspace(0, len(data) / FS, data.shape[0] - 1), data[1:, i])
        # plt.ylim(-10000, 10000)
        plt.title(f"Ch {i}")

    plt.show()

    # oscilloscope = RealTimeOscilloscope(stream_client, 64, FS, 3, 30)
    # oscilloscope.run()
