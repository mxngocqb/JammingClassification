from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt

sdr = _bladerf.BladeRF()

print("Device info:", _bladerf.get_device_list()[0])
print("libbladeRF version:", _bladerf.version()) # v2.5.0
print("Firmware version:", sdr.get_fw_version()) # v2.4.0
print("FPGA version:", sdr.get_fpga_version())   # v0.15.0

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0)) # give it a 0 or 1
print("sample_rate_range:", rx_ch.sample_rate_range)
print("bandwidth_range:", rx_ch.bandwidth_range)
print("frequency_range:", rx_ch.frequency_range)
print("gain_modes:", rx_ch.gain_modes)
print("manual gain range:", sdr.get_gain_range(_bladerf.CHANNEL_RX(0))) # ch 0 or 1

sample_rate = 10e6
center_freq = 1575.42e6
gain = 60 # -15 to 60 dB
num_samples = int(1e5)

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate/2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

# Setup synchronous stream
sdr.sync_config(
    layout = _bladerf.ChannelLayout.RX_X1, # or RX_X2
    fmt = _bladerf.Format.SC16_Q11,        # int16s
    num_buffers    = 16,
    buffer_size    = 8192,
    num_transfers  = 8,
    stream_timeout = 3500
)

# Create receive buffer
bytes_per_sample = 4 # 2 byte I + 2 byte Q (int16)
buf = bytearray(1024 * bytes_per_sample)

# Enable module
print("Starting receive")
rx_ch.enable = True

# Receive loop
x = np.zeros(num_samples, dtype=np.complex64) # storage for IQ samples
num_samples_read = 0
while True:
    if num_samples > 0 and num_samples_read == num_samples:
        break
    elif num_samples > 0:
        num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
    else:
        num = len(buf) // bytes_per_sample

    sdr.sync_rx(buf, num)  # Read into buffer

    samples = np.frombuffer(buf, dtype=np.int16)
    samples = samples[0::2] + 1j * samples[1::2]  # I + jQ

    x[num_samples_read:num_samples_read+num] = samples[0:num]
    num_samples_read += num

print("Stopping")
rx_ch.enable = False
print(x[0:10])      # first 10 IQ samples
print("max |x| =", np.max(np.abs(x)))  # check overload

# ============================
# 5. VẼ TÍN HIỆU HÌNH SIN THEO THỜI GIAN (KÊNH I)
# ============================

# Trục thời gian (giây)
t = np.arange(len(x)) / sample_rate

# Lấy một đoạn ngắn cho dễ nhìn (vd: 2000 mẫu đầu)
N_plot = min(2000, len(x))
t_plot = t[:N_plot]
x_plot = x[:N_plot]

# Kênh I (phần thực) – đây là cái “hình sin” baseband
y = x_plot.real

plt.figure(figsize=(10, 4))
plt.plot(t_plot * 1e3, y)  # *1e3 để đổi sang ms nếu muốn
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude (I)")
plt.title("Time-domain I channel (expected sinusoid if single tone)")
plt.grid(True)
plt.tight_layout()
plt.show()
