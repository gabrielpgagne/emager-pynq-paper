import time
import numpy as np
import threading
import logging as log
from scipy import signal

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt

import emager_py.streamers as streamers
import globals as g


class RealTimeOscilloscope:
    def __init__(
        self,
        streamer: streamers.EmagerStreamerInterface,
        n_ch: int,
        signal_fs: float,
        accumulate_t: float,
        refresh_rate: float,
    ):
        """
        Create the Oscilloscope.

        - streamer: implements read() method, returning a (n_samples, n_ch) array. Must return (0, n_ch) when no samples are available.
        - n_ch: number of "oscilloscope channels"
        - signal_fs: Signal sample rate [Hz]
        - accumulate_t: x-axis length [s]
        - refresh_rate: oscilloscope refresh rate [Hz]
        """
        self.streamer = streamer
        self.n_ch = n_ch
        self.data_points = int(accumulate_t * signal_fs)
        self.samples_per_refresh = signal_fs // refresh_rate

        print(
            f"Data points: {self.data_points}, samples per refresh: {self.samples_per_refresh}"
        )

        # Create a time axis
        self.t = np.linspace(0, accumulate_t, self.data_points)

        # Initialize the data buffer for each signal
        self.data = [np.zeros(self.data_points) for _ in range(n_ch)]

        # Create the application
        self.app = QApplication([])

        # Create a window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle("Real-Time Oscilloscope")
        self.win.setBackground(QtGui.QColor(255, 255, 255))  # white: (255, 255, 255)

        # Define the number of rows and columns in the grid
        num_rows = 16

        # Create plots for each signal
        self.plots = []
        for i in range(n_ch):
            row = i % num_rows
            col = i // num_rows
            p = self.win.addPlot(row=row, col=col)
            p.setYRange(-10000, 10000)
            p.getAxis("left").setStyle(
                showValues=False
            )  # Remove axis title, keep axis lines
            p.getAxis("bottom").setStyle(
                showValues=False
            )  # Remove axis title, keep axis lines
            graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color="r", width=2))
            self.plots.append(graph)

        # Set up a QTimer to update the plot at the desired refresh rate
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // refresh_rate)
        self.t2 = time.time()
        self.win.show()

        self.timestamp = time.time()
        self.t0 = time.time()
        self.tot_samples = 0

        self.notch = signal.iirnotch(60, 30, signal_fs)

    def update(self):
        # Fetch available data
        new_data = []
        while True:
            tmp_data = self.streamer.read()
            if len(tmp_data) == 0:
                # no more samples ready
                break
            new_data.append(tmp_data)
        new_data = np.array(new_data).reshape(-1, self.n_ch).T
        nb_pts = new_data.shape[1]
        if nb_pts == 0:
            return

        new_data = signal.filtfilt(self.notch[0], self.notch[1], new_data, axis=1)

        self.tot_samples += nb_pts
        t = time.time()
        log.info(
            f"(dt={t - self.timestamp:.3f}) Average fs={self.tot_samples / (t - self.t0):.3f}"
        )
        self.timestamp = t

        for i, plot_item in enumerate(self.plots):
            self.data[i] = np.roll(self.data[i], -nb_pts)  # Shift the data
            # self.data[i][-nb_pts:] = signal.decimate(new_data[i],2)  # Add new data point
            self.data[i][-nb_pts:] = new_data[i]
            plot_item.setData(self.t, self.data[i])

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    import emager_py.emager_redis as er
    import emager_py.finn.remote_operations as ro

    import matplotlib.pyplot as plt

    FS = 1000
    BATCH = 25
    HOST = g.PYNQ_HOSTNAME

    r = er.EmagerRedis(HOST)
    r.set_sampling_params(FS, BATCH, 100000000)
    r.set_rhd_sampler_params(
        low_bw=15,
        hi_bw=350,
        # en_dsp=1,
        bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
    )
    r.clear_data()
    c = ro.connect_to_pynq(hostname=HOST)
    t = threading.Thread(
        target=ro.run_remote_finn,
        args=(c, ro.DEFAULT_EMAGER_PYNQ_PATH, "rhd_sampler"),
    ).start()

    print("Starting client and oscilloscope...")
    stream_client = streamers.RedisStreamer(HOST, False)

    data = []
    t0 = time.time()
    while len(data) < 5 * FS:
        new_data = stream_client.read()
        if len(new_data) == 0:
            continue
        if len(data) == 0:
            t0 = time.time()
        data.extend(new_data)
        print(len(data))

    true_fs = len(data) / (time.time() - t0)
    print(f"Elapsed time: {time.time() - t0:.3f} s")
    print(f"Actual sampling rate: {true_fs:.3f} Hz")

    data = np.array(data).reshape(-1, 64)
    noise_floor = np.sqrt(np.mean((data - np.mean(data)) ** 2))
    print(f"Noise floor: {noise_floor:.2f}")

    # filt = signal.iirnotch(60, 10, true_fs)
    # data = signal.filtfilt(filt[0], filt[1], data, axis=0)

    for i in range(16):
        for j in range(4):
            plt.subplot(4, 16, 16 * j + i + 1)
            plt.plot(data[:, 16 * j + i])
    from scipy import fft

    for i in range(10):
        data0 = data[:, i]
        y = fft.fft(data0)[: len(data0) // 2]
        f = fft.fftfreq(len(data0), 1 / FS)[: len(data0) // 2]
        plt.figure()
        plt.plot(f, 20 * np.log(np.abs(y)))

    plt.show()

    # oscilloscope = RealTimeOscilloscope(stream_client, 64, FS, 3, 30)
    # oscilloscope.run()

    c.close()
