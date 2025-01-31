import numpy as np
import threading
import time

from scipy import signal

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QPushButton,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal

from emager_py.emager_redis import EmagerRedis
from emager_py.finn import remote_operations as ro
from emager_py.utils import EMAGER_CHANNEL_MAP


# Worker Thread to Read Serial Data
class RedisReader(QThread):
    data_received = pyqtSignal(np.ndarray)  # Signal to send data to GUI

    def __init__(self, hostname, parent=None):
        super().__init__(parent)
        self.host = hostname
        self.red = EmagerRedis(hostname=hostname)
        self.red.clear_data()
        self.running = True  # Control flag

    def run(self):
        """Continuously read data from redis"""
        self.red.set_sampling_params(1000, 25, 1e6)
        self.red.set_rhd_sampler_params(
            low_bw=15,
            hi_bw=350,
            # en_dsp=1,
            fp_dsp=20,
            bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
        )

        c = ro.connect_to_pynq(hostname=self.host)
        threading.Thread(
            target=ro.run_remote_finn,
            args=(c, ro.DEFAULT_EMAGER_PYNQ_PATH, "rhd_sampler"),
            daemon=True,
        ).start()

        t0 = time.perf_counter()
        n_samples = 0
        data = []
        while self.running:
            new_data = self.red.pop_sample()[0]
            data.extend(new_data)
            if n_samples == 0:
                t0 = time.perf_counter()
            n_samples += len(new_data)
            if len(data) >= 50:
                print(f"Sample rate: {n_samples / (time.perf_counter() - t0):.2f} Hz")
                self.data_received.emit(np.array(data).T)  # Send data to GUI
                data = []

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.quit()
        self.wait()


class DataPlotter(QMainWindow):
    def __init__(
        self,
        hostname: str,
        remap: bool,
    ):
        """
        Create the Oscilloscope.

        - streamer: implements read() method, returning a (n_samples, n_ch) array. Must return (0, n_ch) when no samples are available.
        - n_ch: number of "oscilloscope channels"
        - signal_fs: Signal sample rate [Hz]
        - accumulate_t: x-axis length [s]
        - refresh_rate: oscilloscope refresh rate [Hz]
        """
        super().__init__()

        # Initialize Serial Reader Thread
        self.worker_th = RedisReader(hostname)
        self.worker_th.data_received.connect(self.update_plot)  # Connect signal

        # GUI Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        # Grid layout for multiple subplots
        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        self.start_worker()

        self.central_widget.setLayout(layout)

        self.buffer_size = 5000
        self.data_buffer = np.zeros((64, self.buffer_size))

        if remap:
            self.channel_map = EMAGER_CHANNEL_MAP
        else:
            self.channel_map = list(np.arange(64))

        # Create 64 subplots
        self.plots = []
        self.curves = []
        self.create_plots()

        # Timer for Updating Plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(100)

    def create_plots(self):
        """Create 64 subplots in a grid layout"""
        rows, cols = 4, 16  # Arrange plots in an 8x8 grid
        for r in range(rows):
            for c in range(cols):
                plot_widget = pg.PlotWidget()
                self.grid_layout.addWidget(plot_widget, r, c)  # Add to grid
                plot_widget.getPlotItem().hideAxis("left")  # Hide Y-axis
                plot_widget.getPlotItem().hideAxis("bottom")  # Hide X-axis

                # Set fixed Y-axis limits
                plot_widget.setYRange(-10000, 10000)

                curve = plot_widget.plot(pen="y")  # Yellow line
                plot_widget.setTitle(f"Ch {16 * r + c}")  # Set title
                self.plots.append(plot_widget)
                self.curves.append(curve)

    def start_worker(self):
        """Start the serial thread"""
        if not self.worker_th.isRunning():
            self.worker_th.start()

    def stop_worker(self):
        """Stop the serial thread"""
        self.worker_th.stop()

    def update_plot(self, values):
        """Receive new data (64 values) and update buffer with shape (64, n_samples)"""
        if values.shape[0] == 64:  # Ensure correct data size
            nb_pts = values.shape[1]
            values = values[self.channel_map]
            self.data_buffer = np.roll(self.data_buffer, -nb_pts, axis=1)  # Shift left
            self.data_buffer[:, -nb_pts:] = values  # Insert new data

    def refresh_plot(self):
        """Refresh all 64 plots"""
        for i in range(64):
            self.curves[i].setData(self.data_buffer[i])

    def closeEvent(self, event):
        """Stop thread safely when closing window"""
        self.stop_worker()
        event.accept()


if __name__ == "__main__":
    import emager_py.utils as eutils
    import sys
    import globals as g

    eutils.set_logging()

    app = QApplication(sys.argv)

    window = DataPlotter(g.PYNQ_HOSTNAME, True)

    window.show()
    sys.exit(app.exec())
