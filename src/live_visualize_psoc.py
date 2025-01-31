import time
import numpy as np
import logging as log
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

import serial

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


# Worker Thread to Read Serial Data
class SerialReader(QThread):
    data_received = pyqtSignal(np.ndarray)  # Signal to send data to GUI

    def __init__(self, port, baudrate=1500000, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.running = True  # Control flag

    def run(self):
        """Continuously read data from the serial port"""
        try:
            with serial.Serial(self.port, self.baudrate, timeout=0.1) as ser:
                t0 = time.perf_counter()
                n_samples = 0
                data = []
                while self.running:
                    data_bytes = np.frombuffer(ser.read(128), dtype=np.uint8)
                    new_data = decode_buffer(data_bytes)
                    data.extend(new_data)
                    if n_samples == 0:
                        t0 = time.perf_counter()
                    n_samples += len(new_data)

                    if len(data) >= 50:
                        print(
                            f"Sample rate: {n_samples / (time.perf_counter() - t0):.2f} Hz"
                        )
                        self.data_received.emit(np.array(data).T)  # Send data to GUI
                        data = []
        except serial.SerialException as e:
            print(f"Serial Error: {e}")

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.quit()
        self.wait()


class SerialPlotter(QMainWindow):
    def __init__(
        self,
        port: str,
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
        self.serial_thread = SerialReader(port, 1500000)
        self.serial_thread.data_received.connect(self.update_plot)  # Connect signal

        # GUI Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        # Grid layout for multiple subplots
        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        self.start_serial()

        self.central_widget.setLayout(layout)

        self.buffer_size = 5000
        self.data_buffer = np.zeros((64, self.buffer_size))

        if remap:
            self.channel_map = EMAGER_CHANNEL_MAP
            assert set(self.channel_map) == set(
                np.arange(64)
            ), "Channel mapping does not cover all channels"
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

    def start_serial(self):
        """Start the serial thread"""
        if not self.serial_thread.isRunning():
            self.serial_thread.start()

    def stop_serial(self):
        """Stop the serial thread"""
        self.serial_thread.stop()

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
        self.stop_serial()
        event.accept()


if __name__ == "__main__":
    import emager_py.utils as eutils
    import sys

    eutils.set_logging()

    app = QApplication(sys.argv)

    window = SerialPlotter("/dev/cu.usbmodem1403", True)

    window.show()
    sys.exit(app.exec())
