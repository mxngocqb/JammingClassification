import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.colors import LinearSegmentedColormap

# ===============================
# CONFIG
# ===============================
fs = 200e6              # sample rate (Hz)
T_total = 400e-6        # tổng thời gian mô phỏng (400 µs)
f_min, f_max = -10e6, 10e6   # tần số quét ±10 MHz
T_swp = 10e-6           # chu kỳ quét (Tswp)
b = 1                   # +1 = up-chirp, -1 = down-chirp
Pj = 1.0                # công suất nhiễu
phi_j = 0               # pha ban đầu
noise_power = 0.02      # công suất nhiễu nền

# ===============================
# DEFINE MATLAB PARULA COLORMAP
# ===============================
parula_data = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270),
    (0.2081, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279),
    (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303),
    (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8820),
    (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786),
    (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641),
    (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312),
    (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240),
    (0.0488, 0.5772, 0.8228),
    (0.0343, 0.5977, 0.8199),
    (0.0265, 0.6178, 0.8135),
    (0.0239, 0.6372, 0.8038),
    (0.0231, 0.6560, 0.7914),
    (0.0228, 0.6742, 0.7768),
    (0.0267, 0.6918, 0.7601),
    (0.0384, 0.7088, 0.7423),
    (0.0590, 0.7250, 0.7240),
    (0.0843, 0.7406, 0.7062),
    (0.1133, 0.7556, 0.6885),
    (0.1453, 0.7699, 0.6708),
    (0.1801, 0.7835, 0.6531),
    (0.2178, 0.7964, 0.6353),
    (0.2586, 0.8085, 0.6173),
    (0.3022, 0.8197, 0.5992),
    (0.3482, 0.8300, 0.5809),
    (0.3953, 0.8394, 0.5624),
    (0.4420, 0.8481, 0.5436),
    (0.4871, 0.8559, 0.5244),
    (0.5300, 0.8630, 0.5046),
    (0.5709, 0.8693, 0.4843),
    (0.6091, 0.8749, 0.4635),
    (0.6444, 0.8798, 0.4423),
    (0.6769, 0.8840, 0.4208),
    (0.7066, 0.8875, 0.3990),
    (0.7334, 0.8903, 0.3771),
    (0.7573, 0.8924, 0.3551),
    (0.7783, 0.8938, 0.3330),
    (0.7964, 0.8945, 0.3109),
    (0.8116, 0.8946, 0.2889),
    (0.8240, 0.8940, 0.2669),
    (0.8336, 0.8928, 0.2451),
    (0.8405, 0.8910, 0.2235),
    (0.8449, 0.8885, 0.2022),
    (0.8467, 0.8854, 0.1812),
    (0.8460, 0.8816, 0.1606),
    (0.8428, 0.8772, 0.1406),
    (0.8371, 0.8722, 0.1211),
    (0.8290, 0.8665, 0.1021),
    (0.8186, 0.8602, 0.0838),
    (0.8060, 0.8533, 0.0662),
    (0.7912, 0.8457, 0.0493),
    (0.7743, 0.8375, 0.0332),
    (0.7554, 0.8286, 0.0179),
    (0.7346, 0.8190, 0.0035),
]
parula = LinearSegmentedColormap.from_list("parula", parula_data)

# ===============================
# FUNCTIONS
# ===============================
def chirp_signal(t, f_min, f_max, T_swp, b, Pj, phi_j):
    """
    Linear chirp: j(t) = sqrt(Pj)*exp(j*(2πf_min t + π b (f_max-f_min)/T_swp * t^2 + φ_j))
    """
    k = (f_max - f_min) / T_swp
    phase = 2 * np.pi * (f_min * t + 0.5 * b * k * t**2) + phi_j
    return np.sqrt(Pj) * np.exp(1j * phase)

# ===============================
# GENERATE SIGNAL
# ===============================
N = int(fs * T_total)
t = np.arange(N) / fs
t_mod = np.mod(t, T_swp)  # tạo sawtooth trong khoảng [0, T_swp)

# Chirp lặp liên tục
x_chirp = chirp_signal(t_mod, f_min, f_max, T_swp, b, Pj, phi_j)

# Thêm nhiễu Gaussian nền
noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))
x_total = x_chirp + noise

# ===============================
# SPECTROGRAM
# ===============================
nperseg = 512
noverlap = 500
f, t_spec, S = spectrogram(
    x_total,
    fs=fs,
    window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    return_onesided=False,
    mode='magnitude'
)

# Sắp xếp tần số tăng dần
idx_sort = np.argsort(f)
f = f[idx_sort]
S = S[idx_sort, :]

# Chuyển sang dB
S_db = 20 * np.log10(S + 1e-16)

# ===============================
# PLOT (MATLAB PARULA STYLE)
# ===============================
time_us = t_spec * 1e6
freq_mhz = f / 1e6
vmax = np.max(S_db)
vmin = vmax - 55  # dynamic range 55 dB

fig, ax = plt.subplots(figsize=(6, 3))
pcm = ax.pcolormesh(
    time_us,
    freq_mhz,
    S_db,
    shading='auto',
    cmap=parula,        # 👈 màu giống MATLAB
    vmin=vmin,
    vmax=vmax
)
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Frequency (MHz)")
ax.set_xlim(0, 400)
ax.set_ylim(-20, 20)
ax.set_title("Linear Chirp Interference (MATLAB Parula Colormap)")
fig.colorbar(pcm, ax=ax, label="Power (dB)")
plt.tight_layout()
plt.show()
