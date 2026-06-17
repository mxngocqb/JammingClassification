import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================================
# 1. ARRAY GEOMETRY (4×4 URA)
# ================================================================
N_x = 4
N_y = 4
N = N_x * N_y

fc = 1.57542e9
c = 299792458
lam = c / fc
d = lam / 2        # λ/2 spacing

# Antenna coordinates
xs = np.arange(N_x) * d
ys = np.arange(N_y) * d
xx, yy = np.meshgrid(xs, ys)
positions = np.vstack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()]).T


# ================================================================
# 2. GENERATE 8 GPS SATELLITE DOAs
# ================================================================
np.random.seed(42)
theta_deg_list = np.random.uniform(20, 80, 8)        # elevation
phi_deg_list   = np.random.uniform(0, 360, 8)        # azimuth

print("\n===== TRUE SATELLITE DOAs =====")
for i in range(8):
    print(f"SV{i+1}: Elev={theta_deg_list[i]:.2f}°,  Az={phi_deg_list[i]:.2f}°")

theta_list = np.deg2rad(theta_deg_list)
phi_list   = np.deg2rad(phi_deg_list)


# ================================================================
# 3. STEERING VECTOR
# ================================================================
def steering_vector(theta, phi):
    k = (2*np.pi/lam) * np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    return np.exp(1j * (positions @ k))


# ================================================================
# 4. SIMULATE MULTI-SATELLITE SIGNAL
# ================================================================
num_snapshots = 600
X = np.zeros((N, num_snapshots), dtype=complex)

SNR_dB = 20
SNR = 10**(SNR_dB/10)
noise_var = 1 / SNR

for k in range(num_snapshots):
    x = np.zeros(N, dtype=complex)

    for sv in range(8):
        s = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        a = steering_vector(theta_list[sv], phi_list[sv])
        x += a * s

    noise = np.sqrt(noise_var/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    X[:, k] = x + noise

X = np.nan_to_num(X)

# Covariance matrix
R = (X @ X.conj().T) / num_snapshots
R = np.nan_to_num(R)


# ================================================================
# 5. 2D MUSIC SPECTRUM
# ================================================================
eigvals, eigvecs = LA.eigh(R)
idx = np.argsort(eigvals)[::-1]     # descending
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

K = 8
En = eigvecs[:, K:]                 # Noise subspace

theta_scan = np.deg2rad(np.linspace(0, 90, 91))
phi_scan   = np.deg2rad(np.linspace(0, 360, 361))

P_music = np.zeros((len(theta_scan), len(phi_scan)))

for it, th in enumerate(theta_scan):
    for ip, ph in enumerate(phi_scan):
        a = steering_vector(th, ph)
        denom = np.real(a.conj().T @ En @ En.conj().T @ a)
        denom = max(denom, 1e-12)
        P_music[it, ip] = 1.0 / denom

P_music /= P_music.max()


# ================================================================
# 6. PEAK PICKING (MULTI-BEAM EXTRACTION)
# ================================================================
P_copy = P_music.copy()

theta_step = 90 / (len(theta_scan)-1)
phi_step   = 360 / (len(phi_scan)-1)

dth = int(5 / theta_step)    # ±5° suppression window
dph = int(5 / phi_step)

est_angles = []

for _ in range(8):
    idx = np.unravel_index(np.argmax(P_copy), P_copy.shape)
    it, ip = idx
    est_th = theta_scan[it]
    est_ph = phi_scan[ip]
    est_angles.append((est_th, est_ph))

    # zero out region to avoid same beam
    it_min = max(0, it - dth)
    it_max = min(len(theta_scan), it + dth + 1)
    ip_min = max(0, ip - dph)
    ip_max = min(len(phi_scan), ip + dph + 1)
    P_copy[it_min:it_max, ip_min:ip_max] = 0

est_theta_deg = np.rad2deg([x[0] for x in est_angles])
est_phi_deg   = np.rad2deg([x[1] for x in est_angles])

print("\n===== ESTIMATED DOAs (MUSIC) =====")
for i in range(8):
    print(f"Est{i+1}: Elev={est_theta_deg[i]:.2f}°,  Az={est_phi_deg[i]:.2f}°")


# ================================================================
# 7. BEAM PATTERNS (3D)
# ================================================================
# Precompute ENU mapping grid
Theta_mesh, Phi_mesh = np.meshgrid(theta_scan, phi_scan, indexing="ij")

E = np.cos(Theta_mesh) * np.sin(Phi_mesh)
Nn = np.cos(Theta_mesh) * np.cos(Phi_mesh)
U = np.sin(Theta_mesh)


# ================================================================
# 8. 3D PLOT OF 8 BEAMS
# ================================================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = [
    'red', 'blue', 'green', 'orange', 'purple',
    'cyan', 'yellow', 'magenta'
]

for i, (th0, ph0) in enumerate(est_angles):

    # compute beam pattern for this satellite
    B = np.zeros_like(P_music)
    a0 = steering_vector(th0, ph0)

    for it, th in enumerate(theta_scan):
        for ip, ph in enumerate(phi_scan):
            a = steering_vector(th, ph)
            B[it, ip] = np.abs(a.conj().T @ a0)**2

    B /= B.max()

    # scale ENU mesh
    E_plot = B * E
    N_plot = B * Nn
    U_plot = B * U

    ax.plot_surface(
        E_plot, N_plot, U_plot,
        color=colors[i], alpha=0.55, linewidth=0
    )

    # mark beam peak
    E_peak = np.cos(th0) * np.sin(ph0)
    N_peak = np.cos(th0) * np.cos(ph0)
    U_peak = np.sin(th0)

    ax.scatter(E_peak, N_peak, U_peak,
               color=colors[i], s=60, edgecolors='k',
               label=f"Beam {i+1}")

# formatting
ax.set_title("3D Beam Patterns (8 GPS Satellites, MUSIC)", fontsize=16)
ax.set_xlabel("East")
ax.set_ylabel("North")
ax.set_zlabel("Up")
ax.legend(loc="upper left")

# equal axes
max_range = np.array([
    E.max()-E.min(),
    Nn.max()-Nn.min(),
    U.max()-U.min()
]).max()

mid_x = (E.max()+E.min())/2
mid_y = (Nn.max()+Nn.min())/2
mid_z = (U.max()+U.min())/2

ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2)
ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2)
ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)

plt.show()
