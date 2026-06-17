from bladerf import _bladerf
import numpy as np
import struct
import time

# ===============================
# 1. CONFIG
# ===============================
center_freq = 1575.42e6      # GPS L1
sample_rate = 5e6
gain = 60                    # chỉnh nếu ADC overload
duration_sec = 4             # thời gian ghi chính
warmup_sec = 3               # làm nóng 3 giây

num_samples = int(sample_rate * duration_sec)
outfile = f"gps_l1_{duration_sec}s.bin"

# ===============================
# 2. OPEN DEVICE
# ===============================
sdr = _bladerf.BladeRF()
print("Device:", _bladerf.get_device_list()[0])
print("libbladeRF:", _bladerf.version())
print("Firmware:", sdr.get_fw_version())
print("FPGA:", sdr.get_fpga_version())

rx = sdr.Channel(_bladerf.CHANNEL_RX(0))

# ===============================
# 3. SET PARAMETERS
# ===============================
rx.frequency = center_freq
rx.sample_rate = sample_rate
rx.bandwidth = sample_rate / 2
rx.gain_mode = _bladerf.GainMode.Manual
rx.gain = gain

# ===============================
# 4. CONFIG SYNC RX
# ===============================
sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=4096,
    num_transfers=8,
    stream_timeout=3500
)

# buffer raw int16 I/Q
bytes_per_sample = 4   # int16(I) + int16(Q)
buf = bytearray(4096 * bytes_per_sample)

# ===============================
# 5. WARM-UP 3 SECONDS
# ===============================
print(f"Warm-up RX for {warmup_sec} seconds...")

rx.enable = True
t_start = time.time()

while time.time() - t_start < warmup_sec:
    # đọc bỏ dữ liệu
    sdr.sync_rx(buf, len(buf)//bytes_per_sample)

print("Warm-up done.")

# ===============================
# 6. START RECORDING TO FILE
# ===============================
print(f"Start RX recording for {duration_sec} seconds...")
samples_read = 0

with open(outfile, "wb") as f:
    while samples_read < num_samples:
        num = min(len(buf)//bytes_per_sample, num_samples - samples_read)
        sdr.sync_rx(buf, num)

        f.write(buf[:num * bytes_per_sample])
        samples_read += num

        if samples_read % int(5e6) == 0:
            print("Collected:", samples_read, "/", num_samples)

rx.enable = False
print("DONE. Saved to", outfile)

import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. PARAM
# ===============================
sample_rate = 5e6
center_freq = 1575.42e6
bytes_per_sample = 4      # int16(I)+int16(Q)
infile = "gps_l1_10s.bin"

# ===============================
# 2. LOAD RAW BINARY FILE
# ===============================
print("Loading:", infile)
raw = np.fromfile(infile, dtype=np.int16)

# reshape → [I, Q]
raw_iq = raw.reshape((-1, 2))
iq = raw_iq[:, 0].astype(np.float32) + 1j * raw_iq[:, 1].astype(np.float32)

# Scale Q11 → [-1,1]
iq /= 2048.0

print("Total samples:", len(iq))

# ===============================
# 3. BUILD SPECTROGRAM (PySDR-style)
# ===============================
fft_size = 2048
num_rows = len(iq) // fft_size

print("FFT rows:", num_rows)

spectrogram = np.zeros((num_rows, fft_size))

for i in range(num_rows):
    block = iq[i*fft_size:(i+1)*fft_size]
    fft_block = np.fft.fftshift(np.fft.fft(block))
    spectrogram[i, :] = 10 * np.log10(np.abs(fft_block)**2 + 1e-12)

# Frequency axis for imshow
extent = [
    (center_freq - sample_rate/2) / 1e6,   # min Freq in MHz
    (center_freq + sample_rate/2) / 1e6,   # max Freq in MHz
    len(iq) / sample_rate,                 # max time (sec)
    0                                      # min time
]

# ===============================
# 4. PLOT
# ===============================
plt.figure(figsize=(10, 6))
plt.imshow(
    spectrogram,
    aspect='auto',
    extent=extent,
    cmap='jet'
)

# plt.xlim((center_freq - 1e6)/1e6, (center_freq + 1e6)/1e6)  # zoom ±1 MHz
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.title("Spectrogram (GPS L1)")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
plt.savefig("spectrogram_gps_l1.png", dpi=300)
plt.show()

print("Saved: spectrogram_gps_l1.png")
