from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CHIRP JAMMER FUNCTION
# ============================================================

def chirp_jammer(t, f_min, f_max, T_swp, f_J, b=1, theta_J=0, P_J=1.0):
    K = (f_max - f_min) / T_swp
    phase = 2*np.pi*f_J*t + np.pi*b*K*(t**2) + theta_J
    return np.sqrt(P_J) * np.exp(1j * phase)


# ============================================================
# 2. OPEN DEVICE
# ============================================================

print("Opening bladeRF...")
sdr = _bladerf.BladeRF()


# ============================================================
# 3. PARAMETERS
# ============================================================

sample_rate = 5e6
center_freq = 1575.42e6
# center_freq = 96e6

T_swp = 50e-3
f_min = -150e3
f_max = +150e3
b = 1
P_J = 1.0
theta_J = 0
f_J = f_min

sweep_samples = int(T_swp * sample_rate)
t = np.arange(sweep_samples) / sample_rate

TX_REPEATS = 20
total_samples = sweep_samples * TX_REPEATS

# ============================================================
# 4. GENERATE TX SIGNAL
# ============================================================

chirp = chirp_jammer(t, f_min, f_max, T_swp, f_J, b, theta_J, P_J)
tx_wave = np.tile(chirp, TX_REPEATS).astype(np.complex64)

tx_wave *= 2048.0
iq = np.empty(tx_wave.size * 2, dtype=np.int16)
iq[0::2] = np.int16(np.real(tx_wave))
iq[1::2] = np.int16(np.imag(tx_wave))
tx_buf = iq.tobytes()


# ============================================================
# 5. CONFIG TX STREAM  (BUFFER SET A)
# ============================================================

tx = sdr.Channel(_bladerf.CHANNEL_TX(0))
tx.frequency = center_freq
tx.sample_rate = sample_rate
tx.bandwidth = sample_rate/2
tx.gain = 10

sdr.sync_config(
    layout=_bladerf.ChannelLayout.TX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=32,          # DIFFERENT FROM RX
    buffer_size=2048,        # DIFFERENT FROM RX
    num_transfers=16,
    stream_timeout=3000
)


# ============================================================
# 6. CONFIG RX STREAM  (BUFFER SET B)
# ============================================================

rx = sdr.Channel(_bladerf.CHANNEL_RX(0))
rx.frequency = center_freq
rx.sample_rate = sample_rate
rx.bandwidth = sample_rate/2
rx.gain_mode = _bladerf.GainMode.Manual
rx.gain = 25

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,          # DIFFERENT FROM TX
    buffer_size=4096,        # DIFFERENT FROM TX
    num_transfers=8,
    stream_timeout=3000
)


# ============================================================
# 7. ENABLE CHANNELS
# ============================================================

tx.enable = True
rx.enable = True

print("START FULL-DUPLEX CHIRP...")


# ============================================================
# 8. TX/RX LOOP
# ============================================================

rx_data = np.zeros(total_samples, dtype=np.complex64)
offset = 0

block = sweep_samples

for k in range(TX_REPEATS):

    # TX
    start = k * block * 4
    end   = (k+1) * block * 4
    sdr.sync_tx(tx_buf[start:end], block)

    # RX
    buf = bytearray(block * 4)
    sdr.sync_rx(buf, block)

    raw = np.frombuffer(buf, dtype=np.int16)
    iq_rx = raw[0::2] + 1j * raw[1::2]
    iq_rx = iq_rx.astype(np.complex64) / 2048.0

    rx_data[offset:offset+block] = iq_rx
    offset += block


tx.enable = False
rx.enable = False
print("DONE TX/RX.")


# ============================================================
# 9. SPECTROGRAM
# ============================================================

fft_size = 2048
frames = len(rx_data) // fft_size

spec = np.zeros((fft_size, frames))
window = np.hanning(fft_size)

for i in range(frames):
    seg = rx_data[i*fft_size:(i+1)*fft_size] * window
    S = np.fft.fftshift(np.fft.fft(seg))
    spec[:, i] = 10*np.log10(np.abs(S)**2 + 1e-12)

freqs = np.linspace(center_freq - sample_rate/2,
                    center_freq + sample_rate/2,
                    fft_size)/1e6

plt.figure(figsize=(14, 7))
plt.imshow(spec,
           aspect='auto',
           origin='lower',
           extent=[0, len(rx_data)/sample_rate, freqs[0], freqs[-1]],
           cmap='jet')
plt.title("Correct CHIRP Sweep – Full-Duplex bladeRF")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.colorbar()
plt.savefig("chirp.png")
plt.show()
