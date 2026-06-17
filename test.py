"""
bladerf_qt_realtime_spectrogram.py
──────────────────────────────────
- Process A: bladeRF reader
- Process B: plotter lưu BMP
- Main Qt app: mỗi 500 ms load ảnh mới nhất và show lên cửa sổ

Cài:
    pip install pyqt5 numpy matplotlib pillow

Chạy:
    python bladerf_qt_realtime_spectrogram.py
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from datetime import datetime

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("Agg")
from PIL import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
)

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SAMPLE_RATE  = 60e6
CENTER_FREQ  = 1575.42e6
GAIN         = 20

BUFFER_SIZE  = 8192
NUM_BUFFERS  = 8

WINDOW_LEN   = 512
NOVERLAP     = 384
NFFT         = 512
FMIN_MHZ     = -15.0
FMAX_MHZ     =  15.0
DYN_RANGE_DB = 55

OUTPUT_DIR   = "./BKDATASET/"
IMG_W        = 640
IMG_H        = 480

THROTTLE_S    = 0.1
QUEUE_MAXSIZE = 32

GUI_REFRESH_MS = 500

# ══════════════════════════════════════════════
# COLORMAP
# ══════════════════════════════════════════════
_COLORS = [
    (0.00, "#000000"),
    (0.20, "#0d0d3a"),
    (0.38, "#1a1a8c"),
    (0.55, "#0055ff"),
    (0.68, "#00ccff"),
    (0.80, "#00ffcc"),
    (0.90, "#aaff00"),
    (0.97, "#ffff00"),
    (1.00, "#ffffff"),
]
_cmap = LinearSegmentedColormap.from_list(
    "sdr_waterfall", [(v, c) for v, c in _COLORS]
)
_LUT = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

# ══════════════════════════════════════════════
# HELPER — spectrogram
# ══════════════════════════════════════════════
def compute_spectrogram(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()

    hop = WINDOW_LEN - NOVERLAP
    win = np.hanning(WINDOW_LEN).astype(np.float32)

    if len(x) < WINDOW_LEN:
        return np.zeros((NFFT, 1), dtype=np.float32)

    n_frm = 1 + (len(x) - WINDOW_LEN) // hop
    SY = np.empty((NFFT, n_frm), dtype=np.complex64)

    for k in range(n_frm):
        s = k * hop
        SY[:, k] = np.fft.fft(x[s:s + WINDOW_LEN] * win, n=NFFT)

    SY = np.fft.fftshift(SY, axes=0)
    FY = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / SAMPLE_RATE)) / 1e6

    mask = (FY >= FMIN_MHZ) & (FY <= FMAX_MHZ)
    power_db = 10 * np.log10(np.abs(SY[mask]) ** 2 + 1e-12)
    return power_db


# ══════════════════════════════════════════════
# HELPER — render BMP
# ══════════════════════════════════════════════
def render_bmp(power_db: np.ndarray) -> Image.Image:
    vmax = np.percentile(power_db, 99.5)
    vmin = vmax - DYN_RANGE_DB

    denom = max(vmax - vmin, 1e-6)
    idx = np.clip((power_db - vmin) / denom, 0.0, 1.0)
    idx = (idx * 255).astype(np.uint8)
    idx = idx[::-1, :]

    rgb = _LUT[idx]
    return Image.fromarray(rgb, mode="RGB").resize((IMG_W, IMG_H), Image.BILINEAR)


# ══════════════════════════════════════════════
# PROCESS A — reader
# ══════════════════════════════════════════════
def reader_process(queue: mp.Queue, stop_event: mp.Event):
    from bladerf import _bladerf

    sdr = _bladerf.BladeRF()
    rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))

    rx_ch.frequency = int(CENTER_FREQ)
    rx_ch.sample_rate = int(SAMPLE_RATE)
    rx_ch.bandwidth = int(SAMPLE_RATE / 2)
    rx_ch.gain_mode = _bladerf.GainMode.Manual
    rx_ch.gain = GAIN

    sdr.sync_config(
        layout=_bladerf.ChannelLayout.RX_X1,
        fmt=_bladerf.Format.SC16_Q11,
        num_buffers=32,
        buffer_size=BUFFER_SIZE,
        num_transfers=16,
        stream_timeout=5000,
    )

    buf = bytearray(BUFFER_SIZE * 4)
    last_push_t = 0.0
    chunks = []

    rx_ch.enable = True
    print(f"[reader] started — push mỗi {THROTTLE_S*1000:.0f} ms")

    try:
        while not stop_event.is_set():
            sdr.sync_rx(buf, BUFFER_SIZE)
            raw = np.frombuffer(buf, dtype=np.int16).copy()
            iq = (
                raw[0::2].astype(np.float32)
                + 1j * raw[1::2].astype(np.float32)
            ) / 2048.0

            chunks.append(iq)
            if len(chunks) > NUM_BUFFERS:
                chunks.pop(0)

            now = time.monotonic()
            if now - last_push_t >= THROTTLE_S and len(chunks) == NUM_BUFFERS:
                frame = np.concatenate(chunks).astype(np.complex64)
                try:
                    queue.put_nowait(frame)
                except Exception:
                    pass
                last_push_t = now

    except Exception as e:
        print(f"[reader] error: {e}")
    finally:
        rx_ch.enable = False
        print("[reader] stopped")


# ══════════════════════════════════════════════
# PROCESS B — plotter
# ══════════════════════════════════════════════
def plotter_process(queue: mp.Queue, stop_event: mp.Event):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_idx = 0
    t0 = time.time()
    cnt = 0

    print(f"[plotter] saving → {OUTPUT_DIR}")

    while not stop_event.is_set():
        try:
            x = queue.get(timeout=2.0)
        except Exception:
            continue

        try:
            power_db = compute_spectrogram(x)
            img = render_bmp(power_db)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(OUTPUT_DIR, f"{ts}_{frame_idx:06d}.bmp")
            img.save(path, format="BMP")

            frame_idx += 1
            cnt += 1

            elapsed = time.time() - t0
            if elapsed >= 5.0:
                print(f"[plotter] {cnt/elapsed:.2f} fps | total: {frame_idx}")
                t0 = time.time()
                cnt = 0
        except Exception as e:
            print(f"[plotter] error: {e}")

    print(f"[plotter] stopped — total saved: {frame_idx}")


# ══════════════════════════════════════════════
# QT MAIN WINDOW
# ══════════════════════════════════════════════
class SpectrogramWindow(QMainWindow):
    def __init__(self, queue: mp.Queue, stop_event: mp.Event, p_reader: mp.Process, p_plotter: mp.Process):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.p_reader = p_reader
        self.p_plotter = p_plotter

        self.setWindowTitle("bladeRF Realtime Spectrogram")
        self.resize(900, 700)

        self.last_shown_path = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.image_label = QLabel("Chưa có ảnh...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: black; color: white;")

        self.info_label = QLabel("Đang chạy...")
        self.info_label.setAlignment(Qt.AlignCenter)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_acquisition)

        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.stop_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(GUI_REFRESH_MS)

    def find_latest_bmp(self):
        if not os.path.isdir(OUTPUT_DIR):
            return None

        files = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith(".bmp")
        ]
        if not files:
            return None

        return max(files, key=os.path.getmtime)

    def update_image(self):
        latest = self.find_latest_bmp()
        if latest is None:
            self.info_label.setText("Chưa có file BMP nào.")
            return

        if latest == self.last_shown_path:
            return

        pixmap = QPixmap(latest)
        if pixmap.isNull():
            self.info_label.setText(f"Không load được ảnh: {os.path.basename(latest)}")
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.last_shown_path = latest
        self.info_label.setText(f"Showing: {os.path.basename(latest)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_shown_path and os.path.exists(self.last_shown_path):
            pixmap = QPixmap(self.last_shown_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)

    def stop_acquisition(self):
        self.info_label.setText("Đang dừng...")
        self.stop_event.set()

        if self.p_reader.is_alive():
            self.p_reader.join(timeout=2.0)
        if self.p_plotter.is_alive():
            self.p_plotter.join(timeout=2.0)

        self.info_label.setText("Đã dừng.")

    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = mp.Event()

    p_reader = mp.Process(
        target=reader_process,
        args=(queue, stop_event),
        daemon=True
    )
    p_plotter = mp.Process(
        target=plotter_process,
        args=(queue, stop_event),
        daemon=True
    )

    p_reader.start()
    p_plotter.start()

    app = QApplication(sys.argv)

    window = SpectrogramWindow(queue, stop_event, p_reader, p_plotter)
    window.show()

    def _signal_handler(sig, frame):
        window.stop_acquisition()
        app.quit()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    exit_code = app.exec_()

    stop_event.set()
    if p_reader.is_alive():
        p_reader.join(timeout=2.0)
    if p_plotter.is_alive():
        p_plotter.join(timeout=2.0)

    sys.exit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()