"""
realtime_inference_qt.py
────────────────────────
- Process A: bladeRF reader
- Main Qt app:
    + lấy IQ frame từ queue
    + tính spectrogram trực tiếp trong RAM
    + render ảnh để hiển thị
    + đưa ảnh vào model MobileNet để inference realtime
- Không lưu file BMP xuống folder

Cài:
    pip install pyqt5 numpy matplotlib pillow torch torchvision

Chạy:
    python realtime_inference_qt.py
"""

import os
import sys
import time
import signal
import queue as pyqueue
import multiprocessing as mp

import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("Agg")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
)


# ══════════════════════════════════════════════
# CONFIG SDR
# ══════════════════════════════════════════════
SAMPLE_RATE  = 60e6
CENTER_FREQ  = 1575.42e6
GAIN         = 20

BUFFER_SIZE  = 8192
NUM_BUFFERS  = 8

THROTTLE_S = 0.01
QUEUE_MAXSIZE = 100

# ══════════════════════════════════════════════
# CONFIG SPECTROGRAM
# ══════════════════════════════════════════════
WINDOW_LEN   = 512
NOVERLAP     = 384
NFFT         = 512
FMIN_MHZ     = -15.0
FMAX_MHZ     =  15.0
DYN_RANGE_DB = 55

IMG_W        = 640
IMG_H        = 480

GUI_REFRESH_MS = 100

# ══════════════════════════════════════════════
# CONFIG MODEL
# ══════════════════════════════════════════════
BACKBONE = "mobilenet_v3_small"   # hoặc "mobilenet_v3_large"
MODEL_NAME = "best_gnss_classifier_mobilenetv3_small.pth"
LABEL_NAME = "label_info.pth"
MODEL_INPUT_SIZE = 224

# nếu muốn giảm tần suất inference để đỡ nặng CPU/GPU
INFER_EVERY_N_FRAMES = 1

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
# DEVICE
# ══════════════════════════════════════════════
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ══════════════════════════════════════════════
# HELPER — spectrogram
# ══════════════════════════════════════════════
def compute_spectrogram(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)

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
    return power_db.astype(np.float32)


# ══════════════════════════════════════════════
# HELPER — render PIL image in RAM
# ══════════════════════════════════════════════
def render_pil_image(power_db: np.ndarray) -> Image.Image:
    vmax = np.percentile(power_db, 99.5)
    vmin = vmax - DYN_RANGE_DB

    denom = max(vmax - vmin, 1e-6)
    idx = np.clip((power_db - vmin) / denom, 0.0, 1.0)
    idx = (idx * 255).astype(np.uint8)

    # đảo trục frequency cho giống waterfall/spectrogram
    idx = idx[::-1, :]

    rgb = _LUT[idx]
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    return img


# ══════════════════════════════════════════════
# HELPER — PIL → QPixmap
# ══════════════════════════════════════════════
def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGB")
    arr = np.array(img)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ══════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════
class SpectrogramClassifier:
    def __init__(self, model_path: str, label_info_path: str, device: torch.device):
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        if not os.path.exists(label_info_path):
            raise FileNotFoundError(f"Không tìm thấy label info: {label_info_path}")

        label_info = torch.load(label_info_path, map_location="cpu")
        self.class_names = label_info["class_names"]
        num_classes = label_info["num_classes"]

        if BACKBONE == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(weights=None)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)

        elif BACKBONE == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large(weights=None)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Backbone không hỗ trợ: {BACKBONE}")

        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Using device:", self.device)
        print("Classes:", self.class_names)
        print("Backbone:", BACKBONE)

    def predict(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_idx = pred_idx.item()
        conf = conf.item()
        probs_np = probs.squeeze(0).detach().cpu().numpy()

        return {
            "pred_class": self.class_names[pred_idx],
            "confidence": conf,
            "probs": probs_np
        }


# ══════════════════════════════════════════════
# PROCESS A — bladeRF reader
# ══════════════════════════════════════════════
def reader_process(data_queue: mp.Queue, stop_event: mp.Event):
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
    chunks = []
    last_push_t = 0.0

    rx_ch.enable = True
    print(f"[reader] started — push mỗi {THROTTLE_S*1000:.0f} ms")

    try:
        while not stop_event.is_set():
            sdr.sync_rx(buf, BUFFER_SIZE)

            raw = np.frombuffer(buf, dtype=np.int16).copy()
            iq = (
                raw[0::2].astype(np.float32) +
                1j * raw[1::2].astype(np.float32)
            ) / 2048.0

            chunks.append(iq)
            if len(chunks) > NUM_BUFFERS:
                chunks.pop(0)

            now = time.monotonic()
            if now - last_push_t >= THROTTLE_S and len(chunks) == NUM_BUFFERS:
                frame = np.concatenate(chunks).astype(np.complex64)

                try:
                    if data_queue.full():
                        try:
                            data_queue.get_nowait()
                        except Exception:
                            pass
                    data_queue.put_nowait(frame)
                except Exception:
                    pass

                last_push_t = now

    except Exception as e:
        print(f"[reader] error: {e}")
    finally:
        rx_ch.enable = False
        print("[reader] stopped")


# ══════════════════════════════════════════════
# QT MAIN WINDOW
# ══════════════════════════════════════════════
class SpectrogramInferenceWindow(QMainWindow):
    def __init__(self, data_queue: mp.Queue, stop_event: mp.Event, p_reader: mp.Process):
        super().__init__()
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.p_reader = p_reader

        self.setWindowTitle("bladeRF Realtime Spectrogram + MobileNet Inference")
        self.resize(1000, 760)

        self.frame_count = 0
        self.last_pil = None
        self.last_pred_text = "Chưa có inference"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, MODEL_NAME)
        label_path = os.path.join(script_dir, LABEL_NAME)

        self.classifier = SpectrogramClassifier(model_path, label_path, DEVICE)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.image_label = QLabel("Chưa có ảnh spectrogram...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 520)
        self.image_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")

        self.pred_label = QLabel("Prediction: ---")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.info_label = QLabel("Đang chờ dữ liệu...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px;")

        btn_row = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_acquisition)
        btn_row.addStretch()
        btn_row.addWidget(self.stop_button)
        btn_row.addStretch()

        layout.addWidget(self.image_label)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.info_label)
        layout.addLayout(btn_row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(GUI_REFRESH_MS)

    def update_frame(self):
        latest_frame = None

        # drain queue để lấy frame mới nhất
        while True:
            try:
                latest_frame = self.data_queue.get_nowait()
            except pyqueue.Empty:
                break
            except Exception:
                break

        if latest_frame is None:
            return

        try:
            power_db = compute_spectrogram(latest_frame)
            pil_img = render_pil_image(power_db)
            self.last_pil = pil_img

            # show ảnh
            pixmap = pil_to_qpixmap(pil_img)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

            # inference
            self.frame_count += 1
            if self.frame_count % INFER_EVERY_N_FRAMES == 0:
                result = self.classifier.predict(pil_img)
                pred_class = result["pred_class"]
                confidence = result["confidence"] * 100.0

                probs_text = " | ".join(
                    f"{name}: {prob*100:.1f}%"
                    for name, prob in zip(self.classifier.class_names, result["probs"])
                )

                self.last_pred_text = f"{pred_class} ({confidence:.2f}%)"
                self.pred_label.setText(f"Prediction: {self.last_pred_text}")
                self.info_label.setText(probs_text)

        except Exception as e:
            self.info_label.setText(f"Lỗi xử lý frame: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_pil is not None:
            pixmap = pil_to_qpixmap(self.last_pil)
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

        self.info_label.setText("Đã dừng.")

    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    data_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = mp.Event()

    p_reader = mp.Process(
        target=reader_process,
        args=(data_queue, stop_event),
        daemon=True
    )
    p_reader.start()

    app = QApplication(sys.argv)

    window = SpectrogramInferenceWindow(data_queue, stop_event, p_reader)
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

    sys.exit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()