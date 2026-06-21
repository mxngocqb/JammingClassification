import os
import time
import math
import signal
import queue as pyqueue
import multiprocessing as mp

import numpy as np
from PIL import Image
import coremltools as ct
import torch
from matplotlib.colors import LinearSegmentedColormap


# ══════════════════════════════════════════════
# CONFIG SDR
# ══════════════════════════════════════════════
SAMPLE_RATE  = 60e6
CENTER_FREQ  = 1575.42e6
GAIN         = 20

BUFFER_SIZE  = 8192
NUM_BUFFERS  = 8

THROTTLE_S = 0.0
QUEUE_MAXSIZE_RAW = 32
QUEUE_MAXSIZE_PRE = 32
QUEUE_MAXSIZE_OUT = 256

# ══════════════════════════════════════════════
# CONFIG SPECTROGRAM
# ══════════════════════════════════════════════
WINDOW_LEN   = 512
NOVERLAP     = 384
NFFT         = 512

FMIN_MHZ     = -15.0
FMAX_MHZ     =  15.0
DYN_RANGE_DB = 55

# render thẳng gần kích thước model để giảm tải
IMG_W        = 640
IMG_H        = 480

# IMG_W        = 224
# IMG_H        = 224

# ══════════════════════════════════════════════
# CONFIG CORE ML
# ══════════════════════════════════════════════
MLPACKAGE_NAME = "resnet18_cpu_ne.mlpackage"
LABEL_NAME     = "label_info_resnet18.pth"
MODEL_INPUT_SIZE = 224

RUN_SECONDS = 60.0
WARMUP_FRAMES = 10
PRINT_EVERY = 20

# số worker
NUM_PREPROC_WORKERS = 1
NUM_INFER_WORKERS = 1

# latest-frame policy cho từng queue
DROP_OLD_RAW = True
DROP_OLD_PRE = True

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

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def safe_put_latest(q: mp.Queue, item, overwrite_counter: mp.Value = None):
    try:
        if q.full():
            try:
                q.get_nowait()
                if overwrite_counter is not None:
                    with overwrite_counter.get_lock():
                        overwrite_counter.value += 1
            except Exception:
                pass
        q.put_nowait(item)
        return True
    except Exception:
        if overwrite_counter is not None:
            with overwrite_counter.get_lock():
                overwrite_counter.value += 1
        return False


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


def render_rgb_array(power_db: np.ndarray) -> np.ndarray:
    vmax = np.percentile(power_db, 99.5)
    vmin = vmax - DYN_RANGE_DB

    denom = max(vmax - vmin, 1e-6)
    idx = np.clip((power_db - vmin) / denom, 0.0, 1.0)
    idx = (idx * 255).astype(np.uint8)
    idx = idx[::-1, :]

    rgb = _LUT[idx]
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_rgb_for_coreml(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] != MODEL_INPUT_SIZE or rgb.shape[1] != MODEL_INPUT_SIZE:
        img = Image.fromarray(rgb, mode="RGB")
        img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)

    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = (x - MEAN) / STD
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


class CoreMLSpectrogramClassifier:
    def __init__(self, mlpackage_path: str, label_info_path: str):
        if not os.path.exists(mlpackage_path):
            raise FileNotFoundError(f"Không tìm thấy mlpackage: {mlpackage_path}")
        if not os.path.exists(label_info_path):
            raise FileNotFoundError(f"Không tìm thấy label info: {label_info_path}")

        label_info = torch.load(label_info_path, map_location="cpu")
        self.class_names = label_info["class_names"]
        self.num_classes = label_info["num_classes"]

        self.model = ct.models.MLModel(
            mlpackage_path,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

        print("Loaded CoreML model:", mlpackage_path)
        print("Input name         :", self.input_name)
        print("Output name        :", self.output_name)
        print("Classes            :", self.class_names)

    def predict_from_input(self, x: np.ndarray):
        t0 = time.perf_counter()
        out = self.model.predict({self.input_name: x})
        t1 = time.perf_counter()

        output_value = out[self.output_name]

        if isinstance(output_value, dict):
            best_key = max(output_value, key=output_value.get)
            pred_idx = int(best_key)
            probs = np.zeros((self.num_classes,), dtype=np.float32)
            for k, v in output_value.items():
                probs[int(k)] = float(v)
        else:
            probs = np.array(output_value).reshape(-1).astype(np.float32)
            pred_idx = int(np.argmax(probs))

        return {
            "pred_idx": pred_idx,
            "pred_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]) if pred_idx < len(probs) else 0.0,
            "probs": probs,
            "inference_time_ms": (t1 - t0) * 1000.0,
        }


# ══════════════════════════════════════════════
# READER PROCESS
# ══════════════════════════════════════════════
def reader_process(raw_queue: mp.Queue,
                   stop_event: mp.Event,
                   stat_generated: mp.Value,
                   stat_enqueued_raw: mp.Value,
                   stat_overwritten_raw: mp.Value):
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
    frame_id = 0

    rx_ch.enable = True
    print(f"[reader] started — throttle={THROTTLE_S}s")

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
                payload = {
                    "frame_id": frame_id,
                    "t_ready": now,
                    "iq": frame,
                }

                with stat_generated.get_lock():
                    stat_generated.value += 1

                ok = safe_put_latest(raw_queue, payload, stat_overwritten_raw)
                if ok:
                    with stat_enqueued_raw.get_lock():
                        stat_enqueued_raw.value += 1

                frame_id += 1
                last_push_t = now

    except Exception as e:
        print(f"[reader] error: {e}")
    finally:
        rx_ch.enable = False
        print("[reader] stopped")


# ══════════════════════════════════════════════
# PREPROCESS WORKER
# raw_queue -> pre_queue
# ══════════════════════════════════════════════
def preproc_worker(worker_id: int,
                   raw_queue: mp.Queue,
                   pre_queue: mp.Queue,
                   stop_event: mp.Event,
                   stat_preproc_done: mp.Value,
                   stat_overwritten_pre: mp.Value):
    print(f"[preproc-{worker_id}] started")
    while not stop_event.is_set():
        try:
            item = raw_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue
        except Exception:
            continue

        try:
            iq_frame = item["iq"]

            t_pre0 = time.perf_counter()
            power_db = compute_spectrogram(iq_frame)
            rgb = render_rgb_array(power_db)
            x = preprocess_rgb_for_coreml(rgb)
            t_pre1 = time.perf_counter()

            out_item = {
                "frame_id": item["frame_id"],
                "t_ready": item["t_ready"],
                "x": x,
                "preproc_time_ms": (t_pre1 - t_pre0) * 1000.0,
            }

            safe_put_latest(pre_queue, out_item, stat_overwritten_pre)

            with stat_preproc_done.get_lock():
                stat_preproc_done.value += 1

        except Exception as e:
            print(f"[preproc-{worker_id}] error: {e}")

    print(f"[preproc-{worker_id}] stopped")


# ══════════════════════════════════════════════
# INFERENCE WORKER
# pre_queue -> result_queue
# ══════════════════════════════════════════════
def infer_worker(worker_id: int,
                 pre_queue: mp.Queue,
                 result_queue: mp.Queue,
                 stop_event: mp.Event,
                 mlpackage_path: str,
                 label_path: str,
                 stat_infer_done: mp.Value,
                 stat_skipped_stale_pre: mp.Value):
    print(f"[infer-{worker_id}] starting...")

    classifier = CoreMLSpectrogramClassifier(mlpackage_path, label_path)

    print(f"[infer-{worker_id}] started")

    while not stop_event.is_set():
        latest = None

        while True:
            try:
                item = pre_queue.get(timeout=0.05 if latest is None else 0.0)
                if latest is not None:
                    with stat_skipped_stale_pre.get_lock():
                        stat_skipped_stale_pre.value += 1
                latest = item
            except pyqueue.Empty:
                break
            except Exception:
                break

        if latest is None:
            continue

        try:
            pred = classifier.predict_from_input(latest["x"])
            t_done = time.monotonic()

            result = {
                "frame_id": latest["frame_id"],
                "t_ready": latest["t_ready"],
                "t_done": t_done,
                "preproc_time_ms": latest["preproc_time_ms"],
                "inference_time_ms": pred["inference_time_ms"],
                "pred_class": pred["pred_class"],
                "confidence": pred["confidence"],
            }

            try:
                result_queue.put_nowait(result)
            except Exception:
                pass

            with stat_infer_done.get_lock():
                stat_infer_done.value += 1

        except Exception as e:
            print(f"[infer-{worker_id}] error: {e}")

    print(f"[infer-{worker_id}] stopped")


# ══════════════════════════════════════════════
# COLLECTOR / BENCHMARK
# result_queue -> metrics
# ══════════════════════════════════════════════
def collector_loop(result_queue: mp.Queue,
                   stop_event: mp.Event,
                   stat_generated: mp.Value,
                   stat_enqueued_raw: mp.Value,
                   stat_overwritten_raw: mp.Value,
                   stat_preproc_done: mp.Value,
                   stat_overwritten_pre: mp.Value,
                   stat_infer_done: mp.Value,
                   stat_skipped_stale_pre: mp.Value):
    processed = 0
    e2e_ms_list = []
    inference_ms_list = []
    preproc_ms_list = []

    run_t0 = time.monotonic()
    warmup_done = False
    last_result = None

    while not stop_event.is_set():
        now = time.monotonic()
        if now - run_t0 >= RUN_SECONDS:
            break

        try:
            item = result_queue.get(timeout=0.2)
        except pyqueue.Empty:
            continue
        except Exception:
            continue

        e2e_ms = (item["t_done"] - item["t_ready"]) * 1000.0
        inf_ms = item["inference_time_ms"]
        pre_ms = item["preproc_time_ms"]

        if processed >= WARMUP_FRAMES:
            e2e_ms_list.append(e2e_ms)
            inference_ms_list.append(inf_ms)
            preproc_ms_list.append(pre_ms)

        processed += 1
        last_result = item

        if (not warmup_done) and processed >= WARMUP_FRAMES:
            warmup_done = True
            print(f"[collector] warmup done after {WARMUP_FRAMES} frames")

        if processed % PRINT_EVERY == 0:
            elapsed = time.monotonic() - run_t0
            fps = processed / max(elapsed, 1e-9)
            print(
                f"[collector] processed={processed} | "
                f"fps={fps:.2f} | "
                f"pred={item['pred_class']} ({item['confidence']*100:.1f}%) | "
                f"pre={pre_ms:.2f} ms | "
                f"coreml={inf_ms:.2f} ms | "
                f"e2e={e2e_ms:.2f} ms"
            )

    total_elapsed = time.monotonic() - run_t0

    generated = stat_generated.value
    enqueued_raw = stat_enqueued_raw.value
    overwritten_raw = stat_overwritten_raw.value
    preproc_done = stat_preproc_done.value
    overwritten_pre = stat_overwritten_pre.value
    infer_done = stat_infer_done.value
    skipped_stale_pre = stat_skipped_stale_pre.value

    dropped_total = overwritten_raw + overwritten_pre + skipped_stale_pre
    dropped_ratio = 100.0 * dropped_total / max(generated, 1)

    throughput = len(e2e_ms_list) / max(total_elapsed, 1e-9)

    result = {
        "processed_total": processed,
        "measured_frames": len(e2e_ms_list),
        "generated_frames": generated,
        "enqueued_raw": enqueued_raw,
        "preproc_done": preproc_done,
        "infer_done": infer_done,
        "overwritten_raw": overwritten_raw,
        "overwritten_pre": overwritten_pre,
        "skipped_stale_pre": skipped_stale_pre,
        "dropped_total": dropped_total,
        "dropped_ratio_percent": dropped_ratio,
        "throughput_fps": throughput,
        "e2e_mean_ms": float(np.mean(e2e_ms_list)) if e2e_ms_list else math.nan,
        "e2e_std_ms": float(np.std(e2e_ms_list)) if e2e_ms_list else math.nan,
        "inf_mean_ms": float(np.mean(inference_ms_list)) if inference_ms_list else math.nan,
        "inf_std_ms": float(np.std(inference_ms_list)) if inference_ms_list else math.nan,
        "pre_mean_ms": float(np.mean(preproc_ms_list)) if preproc_ms_list else math.nan,
        "pre_std_ms": float(np.std(preproc_ms_list)) if preproc_ms_list else math.nan,
        "last_prediction": last_result["pred_class"] if last_result else None,
        "last_confidence": last_result["confidence"] if last_result else None,
        "total_elapsed_s": total_elapsed,
    }
    return result


# ══════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════
def print_report(result):
    print("\n" + "=" * 92)
    print("REAL-TIME END-TO-END PIPELINE PERFORMANCE (PARALLEL)")
    print("=" * 92)
    print(f"Processed total frames         : {result['processed_total']}")
    print(f"Measured frames               : {result['measured_frames']}")
    print(f"Generated frames              : {result['generated_frames']}")
    print(f"Enqueued raw                  : {result['enqueued_raw']}")
    print(f"Preprocess done               : {result['preproc_done']}")
    print(f"Inference done                : {result['infer_done']}")
    print(f"Overwritten raw               : {result['overwritten_raw']}")
    print(f"Overwritten pre               : {result['overwritten_pre']}")
    print(f"Skipped stale pre             : {result['skipped_stale_pre']}")
    print(f"Dropped total                 : {result['dropped_total']}")
    print(f"Dropped-frame ratio           : {result['dropped_ratio_percent']:.2f} %")
    print(f"Throughput                    : {result['throughput_fps']:.2f} frames/s")
    print(f"Preprocess latency            : {result['pre_mean_ms']:.2f} ± {result['pre_std_ms']:.2f} ms/frame")
    print(f"Inference latency             : {result['inf_mean_ms']:.2f} ± {result['inf_std_ms']:.2f} ms/frame")
    print(f"End-to-end latency            : {result['e2e_mean_ms']:.2f} ± {result['e2e_std_ms']:.2f} ms/frame")
    print(f"Last prediction               : {result['last_prediction']}")
    if result["last_confidence"] is not None:
        print(f"Last confidence               : {result['last_confidence']*100:.2f} %")
    print("=" * 92)


def print_latex_table(result):
    print("\nLaTeX table row:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Real-Time End-to-End Pipeline Performance}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Metric} & \textbf{Value} & \textbf{Unit} \\")
    print(r"\midrule")
    print(
        rf"End-to-end latency (mean $\pm$ std) & "
        rf"{result['e2e_mean_ms']:.2f} $\pm$ {result['e2e_std_ms']:.2f} & ms/frame \\"
    )
    print(
        rf"Inference latency (CoreML CPU+ANE) & "
        rf"{result['inf_mean_ms']:.2f} $\pm$ {result['inf_std_ms']:.2f} & ms/frame \\"
    )
    print(
        rf"Throughput & {result['throughput_fps']:.2f} & frames/s \\"
    )
    print(
        rf"Dropped-frame ratio & {result['dropped_ratio_percent']:.2f} & \% \\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{table:realtime_e2e}")
    print(r"\end{table}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlpackage_path = os.path.join(script_dir, MLPACKAGE_NAME)
    label_path = os.path.join(script_dir, LABEL_NAME)

    raw_queue = mp.Queue(maxsize=QUEUE_MAXSIZE_RAW)
    pre_queue = mp.Queue(maxsize=QUEUE_MAXSIZE_PRE)
    result_queue = mp.Queue(maxsize=QUEUE_MAXSIZE_OUT)
    stop_event = mp.Event()

    stat_generated = mp.Value("i", 0)
    stat_enqueued_raw = mp.Value("i", 0)
    stat_overwritten_raw = mp.Value("i", 0)

    stat_preproc_done = mp.Value("i", 0)
    stat_overwritten_pre = mp.Value("i", 0)

    stat_infer_done = mp.Value("i", 0)
    stat_skipped_stale_pre = mp.Value("i", 0)

    processes = []

    p_reader = mp.Process(
        target=reader_process,
        args=(
            raw_queue,
            stop_event,
            stat_generated,
            stat_enqueued_raw,
            stat_overwritten_raw,
        ),
        daemon=True,
    )
    processes.append(p_reader)

    for i in range(NUM_PREPROC_WORKERS):
        p = mp.Process(
            target=preproc_worker,
            args=(
                i,
                raw_queue,
                pre_queue,
                stop_event,
                stat_preproc_done,
                stat_overwritten_pre,
            ),
            daemon=True,
        )
        processes.append(p)

    for i in range(NUM_INFER_WORKERS):
        p = mp.Process(
            target=infer_worker,
            args=(
                i,
                pre_queue,
                result_queue,
                stop_event,
                mlpackage_path,
                label_path,
                stat_infer_done,
                stat_skipped_stale_pre,
            ),
            daemon=True,
        )
        processes.append(p)

    for p in processes:
        p.start()

    def _signal_handler(sig, frame):
        print("\n[main] stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        result = collector_loop(
            result_queue=result_queue,
            stop_event=stop_event,
            stat_generated=stat_generated,
            stat_enqueued_raw=stat_enqueued_raw,
            stat_overwritten_raw=stat_overwritten_raw,
            stat_preproc_done=stat_preproc_done,
            stat_overwritten_pre=stat_overwritten_pre,
            stat_infer_done=stat_infer_done,
            stat_skipped_stale_pre=stat_skipped_stale_pre,
        )
        print_report(result)
        print_latex_table(result)

    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.join(timeout=2.0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()