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
        new_data = np.array(new_data).reshape(-1, self.n_ch)
        nb_pts = len(new_data)
        if nb_pts == 0:
            return

        new_data = signal.filtfilt(self.notch[0], self.notch[1], new_data, 0)
        new_data = new_data.T
        self.tot_samples += nb_pts
        t = time.time()
        log.info(
            f"(dt={t-self.timestamp:.3f}) Average fs={self.tot_samples/(t-self.t0):.3f}"
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
    import emager_py.utils as eutils

    eutils.set_logging()

    FS = 1000
    BATCH = 50

    print("Starting client and oscilloscope...")
    stream_client = streamers.SerialStreamer("/dev/ttyACM0", 1500000)
    while True:
        data = stream_client.read()
        if len(data) == 0:
            continue
        print(data.size)
    # oscilloscope = RealTimeOscilloscope(stream_client, 64, FS, 3, 30)
    # oscilloscope.run()
