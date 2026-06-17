import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


# ============================================================
# 1. GLOBAL SYSTEM PARAMETERS
# ============================================================

lam = 1.0
d = lam / 2

auth_angles_deg = np.array([20, 20, 20, 40, -90, 80, -70, 54, 60])
spoof_angle_deg = 20

auth_angles = np.deg2rad(auth_angles_deg)
spoof_angle = np.deg2rad(spoof_angle_deg)

EA = 1.0
ES = 1.0
sigma_n = 0.05
alpha = 0.8
Ns = 100


# ============================================================
# 2. STEERING VECTOR (Eq.2)
# ============================================================

def steering(N, phi):
    n = np.arange(N)
    return np.exp(-1j * 2*np.pi/lam * d * n * np.sin(phi))


# ============================================================
# 3. POST-DESPREADING SIGNAL (Eq.1)
# ============================================================

def generate_y(N, phi_A, phi_S=None):
    noise = sigma_n * (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)

    aA = steering(N, phi_A)
    xA = np.sqrt(EA) * (np.random.randn() + 1j*np.random.randn())/np.sqrt(2)

    if phi_S is None:
        return aA*xA + noise

    aS = steering(N, phi_S)
    xS = alpha*np.sqrt(ES)*(np.random.randn()+1j*np.random.randn())/np.sqrt(2)

    return aA*xA + aS*xS + noise


# ============================================================
# 4. STABLE SOMP (NO NaN – NO overflow – NO singular)
# ============================================================

def stable_SOMP(Y, A, K=2, sigma_n=0.05):
    N, I = Y.shape
    _, P = A.shape

    Yk = Y.copy()
    Lambda = []

    for _ in range(K):

        # AᴴY — bounded
        C = A.conj().T @ Yk
        C = np.nan_to_num(C, nan=0.0, posinf=1e3, neginf=-1e3)

        # correlation — bounded
        corr = np.sum(np.abs(C), axis=1)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1e3)

        # stop if only noise remains
        if corr.max() < 10 * I * sigma_n:
            break

        pk = int(np.argmax(corr))

        if pk in Lambda:
            break

        Lambda.append(pk)

        # projection using stable pinv
        A_L = A[:, Lambda]
        Pk = A_L @ np.linalg.pinv(A_L)
        Pk = np.nan_to_num(Pk, nan=0.0, posinf=1e3, neginf=-1e3)

        Yk = (np.eye(N) - Pk) @ Yk
        Yk = np.nan_to_num(Yk, nan=0.0, posinf=1e3, neginf=-1e3)

    return Lambda


# ============================================================
# 5. MULTI-PRN DIVERSITY SPOOF DETECTION (Sec. III)
# ============================================================

def multi_prn_detect(DoAs, N):
    eta = 0.5 * np.arcsin(2/N)
    S_idx = []

    for i in range(len(DoAs)):
        close_count = 0
        for j in range(len(DoAs)):
            if i != j and abs(DoAs[i] - DoAs[j]) < eta:
                close_count += 1

        if close_count >= 3:
            S_idx.append(i)

    if len(S_idx) == 0:
        return False, None

    spoof_est = float(np.mean([DoAs[i] for i in S_idx]))
    return True, spoof_est


# ============================================================
# 6. FULL SINGLE DETECTION PIPELINE
# ============================================================

def run_detection(N, spoof_present=True):
    P = 181
    phi_grid = np.linspace(-np.pi/2, np.pi/2, P)
    A = np.stack([steering(N, phi) for phi in phi_grid], axis=1)

    estimated_DoAs = []

    for phi_A in auth_angles:
        Y = np.zeros((N, 8), dtype=complex)
        for t in range(8):
            if spoof_present:
                Y[:, t] = generate_y(N, phi_A, spoof_angle)
            else:
                Y[:, t] = generate_y(N, phi_A, None)

        idx = stable_SOMP(Y, A, K=2, sigma_n=sigma_n)
        if len(idx) > 0:
            estimated_DoAs.append(phi_grid[idx[0]])
        else:
            estimated_DoAs.append(0.0)

    detected, spoof_est = multi_prn_detect(np.array(estimated_DoAs), N)
    return detected, spoof_est, np.array(estimated_DoAs)


# ============================================================
# 7. MMSE BEAMFORMING (Eq.8)
# ============================================================

def mmse_beamforming(N, spoof_DoA_est):
    aS = steering(N, spoof_DoA_est)
    phiA = auth_angles[0]
    aA = steering(N, phiA)

    Kz = sigma_n**2*np.eye(N) + ES*np.outer(aS, aS.conj())

    w = LA.solve(Kz, aA)
    if LA.norm(w) > 0:
        w /= LA.norm(w)

    return w


def measure_SINR(N, w):
    phiA = auth_angles[0]
    y = generate_y(N, phiA, spoof_angle)

    aA = steering(N, phiA)
    aS = steering(N, spoof_angle)

    signal = np.abs(w.conj().T @ (aA*np.sqrt(EA)))**2
    interf = np.abs(w.conj().T @ (aS*np.sqrt(ES)))**2
    noise = sigma_n**2 * LA.norm(w)**2

    return 10*np.log10(signal/(interf+noise))


# ============================================================
# 8. PLOT: DoA DEBUG
# ============================================================

def plot_DoA_debug(N):
    detected, spoof_est, DoAs = run_detection(N, True)

    plt.figure(figsize=(10,4))
    plt.stem(np.rad2deg(DoAs))
    plt.axhline(spoof_angle_deg, color='r', linestyle='--', label="True Spoof")
    if spoof_est is not None:
        plt.axhline(np.rad2deg(spoof_est), color='g', linestyle='--', label="Estimated Spoof")
    plt.title("Estimated DoA for 9 PRNs")
    plt.xlabel("PRN index")
    plt.ylabel("DoA (deg)")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 9. PLOT: ROC CURVE (Fig.4)
# ============================================================

def plot_ROC(N=7, runs=200):
    det_list = []
    fa_list = []

    for _ in range(runs):
        detected, _, _ = run_detection(N, True)
        det_list.append(1 if detected else 0)

    for _ in range(runs):
        detected, _, _ = run_detection(N, False)
        fa_list.append(1 if detected else 0)

    P_det = np.mean(det_list)
    P_fa  = np.mean(fa_list)

    plt.figure(figsize=(6,5))
    plt.plot(P_fa, P_det, 'ro', markersize=12)
    plt.title("ROC Curve (Fig.4)")
    plt.xlabel("False Alarm Probability")
    plt.ylabel("Detection Probability")
    plt.grid(True)
    plt.show()


# ============================================================
# 10. PLOT: SINR vs N (Fig.5)
# ============================================================

def plot_SINR_vs_N(maxN=10):
    Ns = list(range(2, maxN+1))
    SINRs = []

    for N in Ns:
        detected, spoof_est, _ = run_detection(N, True)
        if spoof_est is None:
            SINRs.append(-20)
            continue
        w = mmse_beamforming(N, spoof_est)
        SINR = measure_SINR(N, w)
        SINRs.append(SINR)

    plt.figure(figsize=(7,5))
    plt.plot(Ns, SINRs, 'bo-', label="MMSE Beamforming")
    plt.xlabel("Number of Antennas N")
    plt.ylabel("SINR (dB)")
    plt.title("SINR vs Number of Antennas (Fig.5)")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=== RUNNING FULL GNSS SPOOFING DETECTION ===")

    detected, spoof_est, _ = run_detection(N=7, spoof_present=True)
    print("Spoof detected:", detected)

    if spoof_est is not None:
        print("Estimated Spoof DoA:", np.rad2deg(spoof_est))
    else:
        print("Spoof DoA = None (not detected)")

    # Debug DoA
    plot_DoA_debug(7)

    # ROC Curve
    plot_ROC(N=7, runs=200)

    # SINR vs N
    plot_SINR_vs_N(maxN=10)
