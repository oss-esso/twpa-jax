import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ============================================================
# Constants
# ============================================================

PHI0 = 2.067833848e-15          # Magnetic flux quantum [Wb]
PHI0_REDUCED = PHI0 / (2 * np.pi)
Z0 = 50.0                       # Reference impedance [ohm]


# ============================================================
# Parameters
# ============================================================

@dataclass
class IPMTWPAParams:
    # JJ / line parameters
    Cg: float = 66e-15           # Ground capacitance per cell [F]
    Lj: float = 1.18e-12         # Josephson inductance [H]; user gave 1.18 pH
    Rj: float = 105.0            # Junction shunt resistance [ohm]
    fj: float = 40e9             # Freely chosen Josephson plasma frequency [Hz]

    # Cell counts
    n_cells_j1: int = 2000
    n_cells_j2: int = 2000
    n_cells_pump_between_couplers: int = 2000

    # Pump and coupler
    fp: float = 8.0e9            # Pump frequency [Hz]
    pump_current_ratio: float = 0.35
    coupler_f0: float = 8.0e9
    coupler_coupling_db: float = -14.0

    # Toy nonlinear strength multiplier.
    # Set this to 1.0 for the naive scaling.
    # Increase it to make the toy model visibly amplify.
    nonlinear_strength_scale: float = 25.0

    # Pump propagation direction inside the active Josephson-line phase-matching term.
    #
    # +1 means the pump component inside the active line is treated as co-propagating
    # with the signal.
    #
    # -1 means the pump component inside the active line is treated as truly
    # counter-propagating relative to the signal.
    #
    # In many IPM layouts the pump travels counter-propagating on the pump rail,
    # but the coupled pump phase in each active line must be interpreted from
    # the actual directional-coupler convention. Try both.
    pump_k_sign_in_active_line: int = -1

    # Small numerical floor
    eps: float = 1e-30


# ============================================================
# Basic utilities
# ============================================================

def db_to_power_ratio(db: float) -> float:
    return 10.0 ** (db / 10.0)


def db_to_amplitude_ratio(db: float) -> float:
    return 10.0 ** (db / 20.0)


def safe_db_power(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-300))


def jj_cap_from_plasma_frequency(Lj: float, fj: float) -> float:
    """
    User definition:
        Cj = 1 / (Lj * omega_j^2)

    where omega_j = 2*pi*fj.
    """
    omega_j = 2.0 * np.pi * fj
    return 1.0 / (Lj * omega_j**2)


def josephson_critical_current(Lj: float) -> float:
    """
    Small-signal Josephson inductance:
        Lj = phi0_reduced / Ic

    Therefore:
        Ic = phi0_reduced / Lj
    """
    return PHI0_REDUCED / Lj


# ============================================================
# Linear circuit blocks: JJ unit cell and ABCD utilities
# ============================================================

def jj_series_impedance(omega: complex, Lj: float, Cj: float, Rj: float) -> complex:
    """
    RCSJ-like linearized junction impedance.

    The junction is represented as a parallel combination of:
        Lj, Cj, Rj

    Since the junction sits in the series path of the transmission line,
    its equivalent impedance is used as the series impedance of the cell.
    """
    Y_L = 1.0 / (1j * omega * Lj)
    Y_C = 1j * omega * Cj
    Y_R = 1.0 / Rj if np.isfinite(Rj) and Rj > 0 else 0.0
    Y_total = Y_L + Y_C + Y_R
    return 1.0 / Y_total


def shunt_admittance_ground_cap(omega: complex, Cg: float) -> complex:
    return 1j * omega * Cg


def abcd_series_impedance(Z: complex) -> np.ndarray:
    return np.array(
        [[1.0 + 0j, Z],
         [0.0 + 0j, 1.0 + 0j]],
        dtype=complex,
    )


def abcd_shunt_admittance(Y: complex) -> np.ndarray:
    return np.array(
        [[1.0 + 0j, 0.0 + 0j],
         [Y, 1.0 + 0j]],
        dtype=complex,
    )


def jj_unit_cell_abcd(freq: float, params: IPMTWPAParams) -> np.ndarray:
    """
    Symmetric T-cell:

        series JJ/2 -- shunt Cg -- series JJ/2

    This is a linearized cell. The nonlinear parametric part is handled
    separately by josephson_parametric_block().
    """
    omega = 2.0 * np.pi * freq
    Cj = jj_cap_from_plasma_frequency(params.Lj, params.fj)

    Zj = jj_series_impedance(omega, params.Lj, Cj, params.Rj)
    Yg = shunt_admittance_ground_cap(omega, params.Cg)

    return (
        abcd_series_impedance(Zj / 2.0)
        @ abcd_shunt_admittance(Yg)
        @ abcd_series_impedance(Zj / 2.0)
    )


def cascade_abcd(M: np.ndarray, n: int) -> np.ndarray:
    """
    Repeated identical cell cascade.
    """
    if n <= 0:
        return np.eye(2, dtype=complex)
    return np.linalg.matrix_power(M, n)


def abcd_to_s21(M: np.ndarray, z0: float = Z0) -> complex:
    """
    Convert ABCD matrix to S21 for equal reference impedance z0.
    """
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    denom = A + B / z0 + C * z0 + D
    return 2.0 / denom


def cell_phase_from_abcd(M: np.ndarray) -> complex:
    """
    For a periodic cell, cos(theta_cell) = (A + D)/2.

    theta_cell is the Bloch phase per cell. It may become complex in stop bands.
    """
    A, D = M[0, 0], M[1, 1]
    return np.arccos((A + D) / 2.0)


def bloch_phase_per_cell(freq: float, params: IPMTWPAParams) -> complex:
    M = jj_unit_cell_abcd(freq, params)
    return cell_phase_from_abcd(M)


def passive_line_s21(freq: float, n_cells: int, params: IPMTWPAParams) -> complex:
    """
    Passive transmission through a linearized JJ line section.
    """
    M_cell = jj_unit_cell_abcd(freq, params)
    M_total = cascade_abcd(M_cell, n_cells)
    return abcd_to_s21(M_total, Z0)


# ============================================================
# Directional coupler block
# ============================================================

def directional_coupler_coefficients(freq: float, params: IPMTWPAParams) -> dict:
    """
    Simple ideal directional-coupler model.

    The user gave:
        coupling = -14 dB at 8 GHz

    We use constant coupling here. You can later replace this with an actual
    frequency-dependent coupler S-matrix from your coupler geometry.

    Power coupling:
        |kappa|^2 = 10^(-14/10)

    Through:
        |t|^2 = 1 - |kappa|^2
    """
    power_coupling = db_to_power_ratio(params.coupler_coupling_db)
    kappa_mag = np.sqrt(power_coupling)
    through_mag = np.sqrt(max(0.0, 1.0 - power_coupling))

    # Common ideal directional-coupler phase convention.
    kappa = 1j * kappa_mag
    through = through_mag + 0j

    return {
        "kappa": kappa,
        "through": through,
        "power_coupling": power_coupling,
    }


def coupler_signal_idler_block(freq_s: float, freq_i: float, params: IPMTWPAParams) -> np.ndarray:
    """
    Two-mode signal/idler block for the top signal rail.

    v = [a_s, a_i^*]^T

    C_k = [[t_s, 0],
           [0,   t_i^*]]

    In this toy model t_s and t_i come from the ideal through coefficient
    of the coupler. Later these should come from the actual 4-port coupler S-matrix.
    """
    t_s = directional_coupler_coefficients(freq_s, params)["through"]
    t_i = directional_coupler_coefficients(freq_i, params)["through"]

    return np.array(
        [[t_s, 0.0 + 0j],
         [0.0 + 0j, np.conj(t_i)]],
        dtype=complex,
    )


# ============================================================
# Counter-propagating pump model
# ============================================================

def pump_local_amplitudes_counterprop(params: IPMTWPAParams) -> dict:
    """
    Pump enters from the right at Port 4 and travels left.

    It reaches Coupler 2 first, then propagates along the pump rail
    to Coupler 1.

    Returns the complex local pump amplitudes coupled into Josephson
    line 2 and line 1.

    This is a simplified model:
        A_p2 = kappa * A_p,in
        A_p1 = kappa * through * exp(-i theta_pump_between) * A_p,in

    The pump phase between couplers is estimated using the same JJ unit-cell
    Bloch phase just to provide a phase accumulation. In your real device,
    this should be the pump-rail propagation constant from the actual coupler
    / CPW geometry.
    """
    fp = params.fp
    coupler = directional_coupler_coefficients(fp, params)
    kappa = coupler["kappa"]
    through = coupler["through"]

    # Input pump amplitude in arbitrary normalized units.
    A_p_in = 1.0 + 0j

    theta_p_cell = bloch_phase_per_cell(fp, params)
    theta_between = params.n_cells_pump_between_couplers * theta_p_cell

    # Pump reaches C2 first.
    A_p2 = kappa * A_p_in

    # Remaining pump goes from C2 to C1.
    A_p_at_c1_bottom = through * np.exp(-1j * theta_between) * A_p_in
    A_p1 = kappa * A_p_at_c1_bottom

    return {
        "A_p1": A_p1,
        "A_p2": A_p2,
        "theta_between": theta_between,
    }


# ============================================================
# Josephson parametric block
# ============================================================

def josephson_parametric_block(
    freq_s: float,
    freq_p: float,
    n_cells: int,
    local_pump_complex_amplitude: complex,
    params: IPMTWPAParams,
) -> np.ndarray:
    """
    Small-signal two-mode Josephson-line parametric block.

    This is a toy coupled-mode model. It is not a full HB solve.

    Degenerate 4WM:
        f_i = 2 f_p - f_s

    The block acts on:
        v = [a_s, a_i^*]^T

    with:
        J = [[mu, nu],
             [nu^*, mu^*]]

    The gain coefficient is estimated from:
        kappa_cell ~ beta_p_cell * (Ip/Ic)^2 / 8

    multiplied by nonlinear_strength_scale to make the effect visible.

    Phase mismatch:
        Delta beta = beta_s + beta_i - 2 * pump_sign * beta_p

    pump_sign = +1: pump co-propagating in active line.
    pump_sign = -1: pump counter-propagating in active line.

    If you set pump_sign = -1, the mismatch usually becomes large and
    the gain will drop unless the structure is specifically phase matched
    for that case.
    """
    freq_i = 2.0 * freq_p - freq_s

    if freq_i <= 0:
        return np.eye(2, dtype=complex) * np.nan

    beta_s = bloch_phase_per_cell(freq_s, params)
    beta_i = bloch_phase_per_cell(freq_i, params)
    beta_p = bloch_phase_per_cell(freq_p, params)

    pump_sign = params.pump_k_sign_in_active_line

    delta_beta_cell = beta_s + beta_i - 2.0 * pump_sign * beta_p

    Ic = josephson_critical_current(params.Lj)

    # Pump current ratio is user-controlled. Coupler amplitude controls
    # how much of that pump is locally seen by this line.
    local_pump_scale = np.abs(local_pump_complex_amplitude)
    local_pump_phase = np.angle(local_pump_complex_amplitude)

    Ip_over_Ic = params.pump_current_ratio * local_pump_scale

    # Toy FWM coupling per cell.
    kappa_cell = (
        params.nonlinear_strength_scale
        * beta_p
        * (Ip_over_Ic**2)
        / 8.0
    )

    # Complex gain coefficient.
    g_cell = np.sqrt(kappa_cell**2 - (delta_beta_cell / 2.0) ** 2 + 0j)

    x = g_cell * n_cells

    if np.abs(g_cell) < params.eps:
        # Limit sinh(gN)/g -> N
        sinh_over_g = n_cells
    else:
        sinh_over_g = np.sinh(x) / g_cell

    common_phase = np.exp(1j * delta_beta_cell * n_cells / 2.0)

    mu = common_phase * (
        np.cosh(x)
        - 1j * (delta_beta_cell / 2.0) * sinh_over_g
    )

    # The pump phase enters the idler-generating coefficient.
    # For 4WM the mixing phase scales approximately like 2 phi_p.
    nu = common_phase * (
        1j * kappa_cell * sinh_over_g * np.exp(2j * local_pump_phase)
    )

    # Include passive transmission of the active line itself.
    # This makes the line finite-bandwidth and includes linear loss/reflection
    # approximately through S21. The conjugate is used for the idler component
    # in the [a_s, a_i^*] basis.
    t_s_line = passive_line_s21(freq_s, n_cells, params)
    t_i_line = passive_line_s21(freq_i, n_cells, params)

    mu_eff = t_s_line * mu
    nu_eff = t_s_line * nu

    return np.array(
        [[mu_eff, nu_eff],
         [np.conj(nu_eff), np.conj(t_i_line * mu)]],
        dtype=complex,
    )


# ============================================================
# Full IPM device
# ============================================================

def ipm_twpa_transfer_matrix(freq_s: float, params: IPMTWPAParams) -> np.ndarray:
    """
    Build the full modular IPM-like device:

        signal path:
            C1 -> J1 -> C2 -> J2

        pump path:
            Port 4 -> C2 -> C1 -> Port 3

    The total signal/idler transfer matrix is:

        T = J2 C2 J1 C1
    """
    freq_p = params.fp
    freq_i = 2.0 * freq_p - freq_s

    if freq_i <= 0:
        return np.eye(2, dtype=complex) * np.nan

    pump = pump_local_amplitudes_counterprop(params)

    A_p1 = pump["A_p1"]
    A_p2 = pump["A_p2"]

    C1 = coupler_signal_idler_block(freq_s, freq_i, params)
    C2 = coupler_signal_idler_block(freq_s, freq_i, params)

    J1 = josephson_parametric_block(
        freq_s=freq_s,
        freq_p=freq_p,
        n_cells=params.n_cells_j1,
        local_pump_complex_amplitude=A_p1,
        params=params,
    )

    J2 = josephson_parametric_block(
        freq_s=freq_s,
        freq_p=freq_p,
        n_cells=params.n_cells_j2,
        local_pump_complex_amplitude=A_p2,
        params=params,
    )

    return J2 @ C2 @ J1 @ C1


def ipm_twpa_gain(freq_s: float, params: IPMTWPAParams) -> float:
    """
    Signal gain with no injected idler.

        v_in = [1, 0]^T
        v_out = T v_in
        G_s = |a_s,out|^2 / |a_s,in|^2

    Since a_s,in = 1:
        G_s = |v_out[0]|^2
    """
    T = ipm_twpa_transfer_matrix(freq_s, params)
    v_in = np.array([1.0 + 0j, 0.0 + 0j])
    v_out = T @ v_in
    return np.abs(v_out[0]) ** 2


def ipm_twpa_gain_components(freq_s: float, params: IPMTWPAParams) -> dict:
    """
    Returns total gain and a few diagnostic pieces.
    """
    freq_p = params.fp
    freq_i = 2.0 * freq_p - freq_s

    T = ipm_twpa_transfer_matrix(freq_s, params)

    pump = pump_local_amplitudes_counterprop(params)

    C1 = coupler_signal_idler_block(freq_s, freq_i, params)
    C2 = coupler_signal_idler_block(freq_s, freq_i, params)

    J1 = josephson_parametric_block(
        freq_s, freq_p, params.n_cells_j1, pump["A_p1"], params
    )
    J2 = josephson_parametric_block(
        freq_s, freq_p, params.n_cells_j2, pump["A_p2"], params
    )

    v_in = np.array([1.0 + 0j, 0.0 + 0j])
    v_after_c1 = C1 @ v_in
    v_after_j1 = J1 @ v_after_c1
    v_after_c2 = C2 @ v_after_j1
    v_out = J2 @ v_after_c2

    return {
        "freq_i": freq_i,
        "gain_total": np.abs(v_out[0]) ** 2,
        "gain_after_j1": np.abs(v_after_j1[0]) ** 2,
        "gain_after_j2": np.abs(v_out[0]) ** 2,
        "idler_power_out": np.abs(v_out[1]) ** 2,
        "A_p1": pump["A_p1"],
        "A_p2": pump["A_p2"],
        "T": T,
    }


# ============================================================
# Main plotting script
# ============================================================

def main():
    params = IPMTWPAParams()

    Cj = jj_cap_from_plasma_frequency(params.Lj, params.fj)
    Ic = josephson_critical_current(params.Lj)

    print("=== Toy IPM TWPA parameters ===")
    print(f"Cg = {params.Cg:.3e} F")
    print(f"Lj = {params.Lj:.3e} H")
    print(f"Rj = {params.Rj:.3f} ohm")
    print(f"fj = {params.fj / 1e9:.3f} GHz")
    print(f"Cj = {Cj:.3e} F")
    print(f"Ic = {Ic:.3e} A")
    print(f"Pump frequency = {params.fp / 1e9:.3f} GHz")
    print(f"Coupler power coupling = {params.coupler_coupling_db:.2f} dB")
    print(f"pump_k_sign_in_active_line = {params.pump_k_sign_in_active_line}")
    print(f"nonlinear_strength_scale = {params.nonlinear_strength_scale}")
    print()

    pump = pump_local_amplitudes_counterprop(params)
    print("=== Local pump amplitudes from counter-propagating pump rail ===")
    print(f"|A_p2| = {abs(pump['A_p2']):.4f}, phase = {np.angle(pump['A_p2']):.4f} rad")
    print(f"|A_p1| = {abs(pump['A_p1']):.4f}, phase = {np.angle(pump['A_p1']):.4f} rad")
    print("Pump reaches J2 first, then J1.")
    print()

    # Signal sweep.
    # For degenerate 4WM with fp = 8 GHz:
    #     fi = 2 fp - fs
    # Keep fs below 2fp so idler stays positive.
    f_start = 4.0e9
    f_stop = 12.0e9
    n_points = 401

    freqs = np.linspace(f_start, f_stop, n_points)

    gains = []
    idler_powers = []
    gain_after_j1 = []

    for fs in freqs:
        out = ipm_twpa_gain_components(fs, params)
        gains.append(out["gain_total"])
        idler_powers.append(out["idler_power_out"])
        gain_after_j1.append(out["gain_after_j1"])

    gains = np.array(gains)
    idler_powers = np.array(idler_powers)
    gain_after_j1 = np.array(gain_after_j1)

    gain_db = safe_db_power(gains)
    idler_db = safe_db_power(idler_powers)
    gain_after_j1_db = safe_db_power(gain_after_j1)

    # Also compare with pump_sign = -1 if current run is +1, or vice versa.
    params_alt = IPMTWPAParams(
        Cg=params.Cg,
        Lj=params.Lj,
        Rj=params.Rj,
        fj=params.fj,
        n_cells_j1=params.n_cells_j1,
        n_cells_j2=params.n_cells_j2,
        n_cells_pump_between_couplers=params.n_cells_pump_between_couplers,
        fp=params.fp,
        pump_current_ratio=params.pump_current_ratio,
        coupler_f0=params.coupler_f0,
        coupler_coupling_db=params.coupler_coupling_db,
        nonlinear_strength_scale=params.nonlinear_strength_scale,
        pump_k_sign_in_active_line=-params.pump_k_sign_in_active_line,
    )

    gains_alt = np.array([ipm_twpa_gain(fs, params_alt) for fs in freqs])
    gain_alt_db = safe_db_power(gains_alt)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e9, gain_db, label=f"Signal gain, pump sign {params.pump_k_sign_in_active_line:+d}")
    plt.plot(freqs / 1e9, gain_alt_db, "--", label=f"Signal gain, pump sign {-params.pump_k_sign_in_active_line:+d}")
    plt.plot(freqs / 1e9, gain_after_j1_db, ":", label="After first Josephson line")
    plt.xlabel("Signal frequency $f_s$ [GHz]")
    plt.ylabel("Gain [dB]")
    plt.title("Toy IPM TWPA gain: $C_1 \\rightarrow J_1 \\rightarrow C_2 \\rightarrow J_2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ipm_twpa_toy_gain.png", dpi=200)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e9, idler_db, label="Output idler component")
    plt.xlabel("Signal frequency $f_s$ [GHz]")
    plt.ylabel("Idler output power for unit signal input [dB]")
    plt.title("Toy IPM TWPA generated idler")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ipm_twpa_toy_idler.png", dpi=200)

    plt.show()

    print("Saved:")
    print("  ipm_twpa_toy_gain.png")
    print("  ipm_twpa_toy_idler.png")


if __name__ == "__main__":
    main()