import os, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
out_root = "dataset_clean"

# ✅ Chọn lớp cần sinh (tuỳ chọn)
# Ví dụ: classes_to_generate = ["no_jam"]
classes_to_generate = ["no_jam"]
# hoặc ["am", "fm", "chirp", "no_jam"]
# classes_to_generate = ["am", "fm", "chirp", "no_jam"]

n_per_class = 10000
fs = 1_000_000
duration = 0.1
samples = int(fs * duration)
noise_bg_power = 0.001
enable_fading = True
fade_speed_min, fade_speed_max = 50, 2e3
fade_depth = 0.4
rng = np.random.default_rng(1234)

nperseg = 1024
noverlap = 900
v_dynamic_range_db = 55
dpi = 100
img_size = (2.5, 2.5)

# -----------------------
# PARULA COLORMAP (MATLAB)
# -----------------------
parula_data = [
    (0.2081,0.1663,0.5292),(0.2116,0.1898,0.5777),(0.2123,0.2138,0.6270),
    (0.2081,0.2386,0.6771),(0.1959,0.2645,0.7279),(0.1707,0.2919,0.7792),
    (0.1253,0.3242,0.8303),(0.0591,0.3598,0.8683),(0.0117,0.3875,0.8820),
    (0.0060,0.4086,0.8828),(0.0165,0.4266,0.8786),(0.0329,0.4430,0.8720),
    (0.0498,0.4586,0.8641),(0.0629,0.4737,0.8554),(0.0723,0.4887,0.8467),
    (0.0779,0.5040,0.8384),(0.0793,0.5200,0.8312),(0.0749,0.5375,0.8263),
    (0.0641,0.5570,0.8240),(0.0488,0.5772,0.8228),(0.0343,0.5977,0.8199),
    (0.0265,0.6178,0.8135),(0.0239,0.6372,0.8038),(0.0231,0.6560,0.7914),
    (0.0228,0.6742,0.7768),(0.0267,0.6918,0.7601),(0.0384,0.7088,0.7423),
    (0.0590,0.7250,0.7240),(0.0843,0.7406,0.7062),(0.1133,0.7556,0.6885),
    (0.1453,0.7699,0.6708),(0.1801,0.7835,0.6531),(0.2178,0.7964,0.6353),
    (0.2586,0.8085,0.6173),(0.3022,0.8197,0.5992),(0.3482,0.8300,0.5809),
    (0.3953,0.8394,0.5624),(0.4420,0.8481,0.5436),(0.4871,0.8559,0.5244),
    (0.5300,0.8630,0.5046),(0.5709,0.8693,0.4843),(0.6091,0.8749,0.4635),
    (0.6444,0.8798,0.4423),(0.6769,0.8840,0.4208),(0.7066,0.8875,0.3990),
    (0.7334,0.8903,0.3771),(0.7573,0.8924,0.3551),(0.7783,0.8938,0.3330),
    (0.7964,0.8945,0.3109),(0.8116,0.8946,0.2889),(0.8240,0.8940,0.2669),
    (0.8336,0.8928,0.2451),(0.8405,0.8910,0.2235),(0.8449,0.8885,0.2022),
    (0.8467,0.8854,0.1812),(0.8460,0.8816,0.1606),(0.8428,0.8772,0.1406),
    (0.8371,0.8722,0.1211),(0.8290,0.8665,0.1021),(0.8186,0.8602,0.0838),
    (0.8060,0.8533,0.0662),(0.7912,0.8457,0.0493),(0.7743,0.8375,0.0332),
    (0.7554,0.8286,0.0179),(0.7346,0.8190,0.0035)
]
parula = LinearSegmentedColormap.from_list("parula", parula_data)

# -----------------------
# SIGNAL HELPERS
# -----------------------
def fading(n, fs):
    f = rng.uniform(fade_speed_min, fade_speed_max)
    env = 1 + fade_depth * np.sin(2*np.pi*f*np.arange(n)/fs)
    env += 0.05*rng.standard_normal(n)
    return np.clip(env, 0.1, 2.0)

def am_mod(n, fs, fc, fm, idx=0.8):
    t = np.arange(n)/fs
    return (1 + idx*np.sin(2*np.pi*fm*t)) * np.exp(1j*2*np.pi*fc*t)

def fm_mod(n, fs, fc, fm, dev):
    t = np.arange(n)/fs
    phase = 2*np.pi*(fc*t + dev*np.sin(2*np.pi*fm*t)/(2*np.pi*fm))
    return np.exp(1j*phase)

def chirp_mod(n, fs, f0, f1, Ts, b=1):
    t = np.arange(n)/fs
    tm = np.mod(t, Ts)
    k = (f1 - f0)/Ts
    ph = 2*np.pi*(f0*tm + 0.5*b*k*tm**2)
    return np.exp(1j*ph)

def no_jam_signal(n, fs):
    """Tín hiệu không nhiễu: chỉ noise + fading"""
    if enable_fading:
        fad = fading(n, fs)
    else:
        fad = np.ones(n)
    noise = np.sqrt(noise_bg_power/2)*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
    x = fad * noise
    return x

# -----------------------
# GENERATE DATA
# -----------------------
os.makedirs(out_root, exist_ok=True)
for cls in classes_to_generate:
    os.makedirs(os.path.join(out_root, cls), exist_ok=True)

labels_path = os.path.join(out_root, "labels.csv")
with open(labels_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["path", "label"])

    for cls in classes_to_generate:
        print(f"🔹 Generating class: {cls}")
        for i in tqdm(range(n_per_class), desc=f"{cls:>7s}"):
            if enable_fading:
                fad = fading(samples, fs)
            else:
                fad = np.ones(samples)

            if cls == "am":
                fc = rng.uniform(-200e3, 200e3)
                fm_hz = rng.uniform(50, 200)
                sig = am_mod(samples, fs, fc, fm_hz)
            elif cls == "fm":
                fc = rng.uniform(-200e3, 200e3)
                fm_hz = rng.uniform(100, 400)
                dev = rng.uniform(1e3, 40e3)
                sig = fm_mod(samples, fs, fc, fm_hz, dev)
            elif cls == "chirp":
                f0 = rng.uniform(-500e3, -100e3)
                f1 = rng.uniform(100e3, 500e3)
                Ts = rng.uniform(5e-3, 20e-3)
                sig = chirp_mod(samples, fs, f0, f1, Ts, b=rng.choice([1, -1]))
            elif cls == "no_jam":
                sig = no_jam_signal(samples, fs)
            else:
                raise RuntimeError(f"Unknown signal type {cls}")

            if cls != "no_jam":  # chỉ thêm noise cho các loại nhiễu
                noise = np.sqrt(noise_bg_power/2)*(rng.standard_normal(samples) + 1j*rng.standard_normal(samples))
                x = fad * sig + noise
            else:
                x = sig  # no_jam đã bao gồm noise

            f, t_spec, Sxx = spectrogram(x, fs=fs, window='hann',
                                         nperseg=nperseg, noverlap=noverlap,
                                         return_onesided=False, mode='magnitude')
            idx = np.argsort(f)
            f = f[idx]; Sxx = Sxx[idx, :]
            S_db = 20*np.log10(Sxx + 1e-12)
            vmax = S_db.max()
            vmin = vmax - v_dynamic_range_db

            fig, ax = plt.subplots(figsize=img_size)
            ax.axis("off")
            plt.margins(0)
            plt.subplots_adjust(0, 0, 1, 1)
            ax.imshow(S_db, extent=[0, 1, -1, 1], cmap=parula,
                      aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            out_path = os.path.join(out_root, cls, f"{cls}_{i:04d}.png")
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            writer.writerow([out_path, cls])

print("✅ Dataset generation done:", out_root)
