"""
Physics-inspired resistive-switching (RS) simulator for type-I core-shell
nanoparticles. Generates the SUPERVISED TRAINING LABELS for Phase 4
(XGBoost) -- i.e. this is what stands in for "real device measurements"
since no one has synthesized/measured most of the MP-screened candidates.

Physics used:
  1. SIMMONS TUNNELING -- current through the shell (barrier) as a function
     of applied voltage, barrier height (confinement offset), and barrier
     width (shell thickness). Standard Simmons (1963) general formula.
  2. TRAP CHARGING/DISCHARGING -- carriers tunnel into core-localized traps
     during a voltage sweep, modulating the effective barrier height
     (Coulomb/image-force lowering) -> this is what produces hysteresis
     between the forward and reverse I-V branches.
  3. POOLE-FRENKEL EMISSION -- field-assisted thermal detrapping governs
     how fast trapped charge escapes after the write pulse is removed ->
     this sets the retention time constant.

Independent (swept) parameters, matching the team's original spec:
    dE_LUMO_eV      : electron confinement barrier / trap depth (0.1-1.5 eV)
    dE_HOMO_eV      : hole confinement barrier (0.1-1.5 eV)
    shell_thick_nm  : tunneling barrier width (1-10 nm)
    core_radius_nm  : confinement volume / trap density scaling
    Nt_cm3          : trap density (1e16-1e19 cm^-3)
    eps_shell       : shell dielectric constant (real range ~4-20)
    Vmax_V          : voltage sweep amplitude

Dependent (simulator output) labels:
    hysteresis_window_V  : delta-V between fwd/rev branches at fixed read I
    on_off_ratio          : I_LRS / I_HRS at fixed read voltage
    retention_tau_s        : exponential decay time constant of LRS -> HRS

NOTE: this is a simplified, physically-motivated model -- not a full TCAD
device simulation. It's built to (a) respect known qualitative physics
(larger barrier -> lower current, deeper traps -> longer retention,
thicker shell -> lower ON/OFF current but different tunneling scaling)
and (b) produce internally CONSISTENT, monotonic-where-expected synthetic
data, which is exactly what the earlier broken CSV lacked.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Physical constants (SI units unless noted)
# ---------------------------------------------------------------
Q = 1.602176634e-19       # elementary charge, C
H = 6.62607015e-34        # Planck constant, J.s
HBAR = H / (2 * np.pi)
M0 = 9.1093837015e-31     # free electron mass, kg
EPS0 = 8.8541878128e-12   # vacuum permittivity, F/m
KB = 1.380649e-23         # Boltzmann constant, J/K
T = 300.0                 # operating temperature, K


# ---------------------------------------------------------------
# 1. SIMMONS TUNNELING CURRENT
# ---------------------------------------------------------------
def simmons_current_density(V, phi_b_eV, d_m, m_eff=0.3):
    """
    Simmons (1963) tunneling current density, with TWO regimes handled
    correctly:
      - LOW FIELD (qV < 2*phi_b): direct tunneling, general Simmons formula.
      - HIGH FIELD (qV >= 2*phi_b): the applied bias exceeds the barrier
        height itself -- direct tunneling formula breaks down (phi_minus
        would go negative/undefined). Transport switches to Fowler-Nordheim
        field emission instead.
    This matters a lot here: with barrier heights of only 0.1-1.5 eV
    (the swept confinement range) and a fixed 3.0 V read voltage
    (matching the reference paper), qV/2 = 1.5 eV already meets or
    exceeds the ENTIRE barrier for most of the parameter sweep -- so the
    high-field branch isn't an edge case, it's the dominant regime for
    this device.
    """
    phi_b = phi_b_eV * Q  # J
    m = m_eff * M0
    V = np.asarray(V, dtype=float)
    phi_b_arr = np.broadcast_to(phi_b, V.shape) if V.ndim else phi_b

    E_field = np.abs(V) / d_m

    # low-field (direct tunneling) branch
    phi_minus = phi_b - Q * V / 2
    phi_plus = phi_b + Q * V / 2
    low_field_valid = phi_minus > 0.02 * phi_b  # stay away from the singular edge

    phi_minus_safe = np.maximum(phi_minus, 1e-3 * Q)
    phi_plus_safe = np.maximum(phi_plus, 1e-3 * Q)
    pref = Q / (2 * np.pi * H * d_m**2)
    A = (4 * np.pi * d_m / H) * np.sqrt(2 * m)
    J_direct = pref * (
        phi_minus_safe * np.exp(-A * np.sqrt(phi_minus_safe))
        - phi_plus_safe * np.exp(-A * np.sqrt(phi_plus_safe))
    )

    # high-field (Fowler-Nordheim) branch
    phi_b_eV_safe = np.maximum(phi_b_eV, 1e-3)
    J_FN = (Q**3 * E_field**2) / (8 * np.pi * H * (phi_b_eV_safe * Q)) * np.exp(
        -8 * np.pi * np.sqrt(2 * m) * (phi_b_eV_safe * Q) ** 1.5 / (3 * H * Q * np.maximum(E_field, 1.0))
    )

    J = np.where(low_field_valid, J_direct, J_FN)
    return J  # A/m^2


def device_current(V, phi_b_eV, d_m, area_m2, m_eff=0.3):
    """Current (A) through a device of given cross-sectional area."""
    J = simmons_current_density(V, phi_b_eV, d_m, m_eff)
    return J * area_m2


# ---------------------------------------------------------------
# Calibration constants -- tuned so a reference parameter set (roughly
# matching the CsPbCl3/Cs4PbCl6 system, see CALIBRATION_REFERENCE below)
# reproduces the paper's three known anchor points:
#   - switching threshold ~8.0 V (write pulse magnitude needed for LRS)
#   - write pulse width 10 ms
#   - LRS retention ~40 min (2400 s) before decaying back toward HRS
# These replace the placeholder nu0=1e13 / sigma=1e-19 that produced
# near-instant release across the whole parameter space in the first
# version -- see calibrate.py for the search that produced these values.
# ---------------------------------------------------------------
ATTEMPT_FREQ_NU0 = 50.0       # 1/s, effective attempt frequency -- calibrated
                               # (not the textbook ~1e13 phonon-attempt value)
                               # so that a representative trap depth (~0.3 eV)
                               # gives a near-zero-field release time constant
                               # of ~2400 s (40 min), matching the paper's
                               # observed LRS retention. See CALIBRATION NOTE
                               # below for the derivation.
                               #
                               # CALIBRATION NOTE: rate = nu0*exp(-E/kT).
                               # Target: rate = 1/2400 s^-1 at E = 0.3 eV,
                               # kT(300K) = 0.02585 eV.
                               # nu0 = (1/2400) * exp(0.3/0.02585) = 45.6 Hz.
                               # This being far below the textbook phonon
                               # attempt frequency reflects that this is a
                               # coarse-grained effective model, not a full
                               # phonon-coupling calculation -- the fitted
                               # value absorbs whatever additional physics
                               # (polaronic effects, multi-phonon processes,
                               # etc.) actually slows real trap release.
CAPTURE_CROSS_SECTION_M2 = 5e-17  # per-trap capture cross-section
WRITE_PULSE_WIDTH_S = 0.010       # 10 ms, matches the reference paper


# ---------------------------------------------------------------
# 2. TRAP-MODULATED BARRIER (produces hysteresis)
# ---------------------------------------------------------------
def coulomb_barrier_lowering_eV(n_trap_fraction, N_traps_total, r_core_m, eps_r):
    """
    Barrier lowering due to ACTUAL TRAPPED CHARGE (Coulomb/charging
    effect), evaluated as the electrostatic potential at the core-shell
    interface from a point charge Q_trap at the core center:
        dPhi (eV) = Q_trap / (4*pi*eps_r*eps0*r_core)
    This ties trap density (via N_traps_total) and core size directly
    into the switching strength -- more/denser traps and a smaller core
    both produce a stronger barrier shift, which is the coupling that
    was MISSING in the first version (and in the earlier broken CSV,
    where N_t showed ~zero correlation with the outputs).
    """
    N_traps_eff = np.sqrt(N_traps_total**2 + 1.0)  # smooth floor at ~1, instead
    # of a hard max(N_traps_total, 1.0) -- the hard floor made every density
    # from 1e16 to 1e18 cm^-3 collapse to an IDENTICAL output (all floored
    # to exactly 1), erasing 2 of the 3 swept decades of Nt's real effect.
    # This smooth version approaches N_traps_total for N>>1 and approaches
    # 1 for N<<1, but varies continuously in between instead of stepping.
    Q_trap = Q * n_trap_fraction * N_traps_eff  # Coulombs
    dphi_V = Q_trap / (4 * np.pi * eps_r * EPS0 * max(r_core_m, 1e-10))
    return dphi_V  # numerically eV per electron


def poole_frenkel_release_rate(trap_depth_eV, E_field, eps_r, nu0=ATTEMPT_FREQ_NU0):
    """
    Thermally-assisted detrapping rate (1/s), field-lowered via the
    classic Poole-Frenkel mechanism (applied-field-driven, distinct
    from the Coulomb charging-lowering above -- this is what sets how
    fast charge escapes once written, i.e. retention):
    rate = nu0 * exp(-(trap_depth - dPhi_PF) / (kB*T))

    The effective barrier floor is capped at a modest fraction of the
    nominal trap depth (not an arbitrary near-zero value) -- at the
    field strengths typical of nm-scale shells, uncapped PF lowering
    can numerically erase the entire barrier, producing unphysical
    "instant release" for every parameter combination. Real traps in
    these systems still retain SOME depth even under strong field
    (lattice relaxation / polaronic effects resist full collapse) --
    this cap is a simplified stand-in for that saturation behavior.
    """
    E_field = np.maximum(E_field, 0.0)
    dphi_pf_J = np.sqrt((Q**3) * E_field / (np.pi * eps_r * EPS0))
    dphi_pf_eV = dphi_pf_J / Q
    dphi_pf_eV = np.minimum(dphi_pf_eV, 0.85 * trap_depth_eV)  # cap at 85% collapse
    eff_barrier_eV = np.maximum(trap_depth_eV - dphi_pf_eV, 1e-3)
    rate = nu0 * np.exp(-(eff_barrier_eV * Q) / (KB * T))
    return rate  # 1/s


# ---------------------------------------------------------------
# 3. FULL DEVICE SIMULATION FOR ONE PARAMETER SET
# ---------------------------------------------------------------
def simulate_device(dE_LUMO_eV, dE_HOMO_eV, shell_thick_nm, core_radius_nm,
                     Nt_cm3, eps_shell, Vmax_V, V_read=3.0, area_m2=1e-14,
                     n_points=60, capture_cross_section_m2=CAPTURE_CROSS_SECTION_M2,
                     pulse_width_s=WRITE_PULSE_WIDTH_S):
    """
    Runs a forward+reverse voltage sweep (0 -> Vmax -> 0), tracking
    trap occupation via a simple capture/release rate equation, then
    a retention simulation at V_read after the sweep ends in the LRS.

    Returns a dict of raw traces (for optional inspection/plotting)
    plus the three summary outputs used as ML labels:
        hysteresis_window_V, on_off_ratio, retention_tau_s
    """
    d_m = shell_thick_nm * 1e-9
    r_core_m = core_radius_nm * 1e-9
    core_vol_m3 = (4 / 3) * np.pi * r_core_m ** 3
    Nt_m3 = Nt_cm3 * 1e6                       # cm^-3 -> m^-3
    N_traps_total = Nt_m3 * core_vol_m3        # total trap count in core

    # electron barrier = dE_LUMO (shell CB above core CB)
    phi_b0 = dE_LUMO_eV
    trap_depth = dE_LUMO_eV  # electrons detrap over the same barrier they were captured by

    # --- forward sweep: 0 -> Vmax, trap filling ---
    # Each swept voltage point is treated as its own discrete write pulse
    # of the paper's actual duration (10 ms default), not an arbitrary
    # 1 ms dwell -- this is what lets the threshold-voltage behavior
    # emerge from the model instead of being disconnected from the real
    # experimental protocol.
    V_fwd = np.linspace(1e-3, Vmax_V, n_points)
    n_trap = 0.0  # fraction of traps filled, 0-1
    I_fwd = np.zeros(n_points)
    dt = pulse_width_s

    for i, V in enumerate(V_fwd):
        E_field = V / d_m
        dphi = coulomb_barrier_lowering_eV(n_trap, N_traps_total, r_core_m, eps_shell)
        phi_eff = max(phi_b0 - dphi, 0.05)  # traps lower the barrier via Coulomb charging
        I = device_current(V, phi_eff, d_m, area_m2)
        I_fwd[i] = I

        # capture: per-trap rate = (areal current density / q) * capture cross-section
        # (this is dimensionally a rate, independent of core volume -- the
        # earlier bug divided by core_vol_m3 here, which suppressed
        # capture by ~20 orders of magnitude for realistic nm-scale cores)
        J = abs(I) / area_m2
        capture_rate = (J / Q) * capture_cross_section_m2
        release_rate = poole_frenkel_release_rate(trap_depth, E_field, eps_shell)
        dn = (capture_rate * (1 - n_trap) - release_rate * n_trap) * dt
        n_trap = np.clip(n_trap + dn, 0.0, 1.0)

    n_trap_LRS = n_trap  # trap occupation reached at Vmax -> this is the "written" state

    # --- reverse sweep: Vmax -> 0, traps stay filled (slow release vs sweep) ---
    V_rev = np.linspace(Vmax_V, 1e-3, n_points)
    I_rev = np.zeros(n_points)
    n_trap_r = n_trap_LRS
    for i, V in enumerate(V_rev):
        E_field = V / d_m
        dphi = coulomb_barrier_lowering_eV(n_trap_r, N_traps_total, r_core_m, eps_shell)
        phi_eff = max(phi_b0 - dphi, 0.05)
        I_rev[i] = device_current(V, phi_eff, d_m, area_m2)
        release_rate = poole_frenkel_release_rate(trap_depth, E_field, eps_shell)
        n_trap_r = np.clip(n_trap_r - release_rate * n_trap_r * dt, 0.0, 1.0)

    # --- hysteresis window: delta-V between fwd/rev branches at fixed read current ---
    I_read_level = np.interp(V_read, V_fwd, I_fwd)
    # find V on reverse branch giving the same current level
    # (reverse branch is descending in V, ascending in I as V drops toward 0
    #  only if there's real hysteresis; guard for monotonic edge cases)
    try:
        V_rev_at_same_I = np.interp(I_read_level, I_rev[::-1], V_rev[::-1])
        hysteresis_window_V = abs(V_read - V_rev_at_same_I)
    except Exception:
        hysteresis_window_V = 0.0

    # --- ON/OFF ratio: read current with traps filled (LRS) vs empty (HRS) ---
    dphi_LRS = coulomb_barrier_lowering_eV(n_trap_LRS, N_traps_total, r_core_m, eps_shell)
    phi_LRS = max(phi_b0 - dphi_LRS, 0.05)
    phi_HRS = phi_b0  # pristine, no trapped charge
    I_LRS = device_current(V_read, phi_LRS, d_m, area_m2)
    I_HRS = device_current(V_read, phi_HRS, d_m, area_m2)
    on_off_ratio = abs(I_LRS) / max(abs(I_HRS), 1e-30)

    # --- retention: decay of n_trap between reads, at near-zero resting
    # bias (V_hold), NOT continuously held at V_read=3.0V.
    #
    # This matches the actual experimental protocol (paper's Fig. 3(d)):
    # the device sits largely unbiased between brief, low-disturbance
    # read pulses used only to sample the current -- it isn't held at
    # 3.0V continuously for the full ~40 min retention window. Using
    # V_read's real field here (as the first version did) meant even
    # the "resting" state saw a large field, causing massive PF barrier
    # lowering and a release time of ~0.1s instead of the target ~2400s.
    V_hold = 0.0  # V, true zero-field resting bias between reads (the
                   # device isn't held at any sustained voltage between
                   # brief read pulses in the actual experimental protocol)
    E_hold = V_hold / d_m
    release_rate_hold = poole_frenkel_release_rate(trap_depth, E_hold, eps_shell)
    # exponential decay: n_trap(t) = n_trap_LRS * exp(-release_rate * t)
    # tau (1/e time) is simply 1/release_rate for a pure exponential model
    retention_tau_s = 1.0 / max(release_rate_hold, 1e-30)

    return {
        "hysteresis_window_V": hysteresis_window_V,
        "on_off_ratio": on_off_ratio,
        "log10_on_off_ratio": np.log10(max(on_off_ratio, 1e-30)),
        "retention_tau_s": retention_tau_s,
        "n_trap_LRS_fraction": n_trap_LRS,
        "N_traps_total": N_traps_total,
    }


# ---------------------------------------------------------------
# 4. SYNTHETIC DATASET GENERATION (Latin Hypercube sampling)
# ---------------------------------------------------------------
PARAM_RANGES = {
    "dE_LUMO_eV":      (0.1, 1.5),
    "dE_HOMO_eV":       (0.1, 1.5),
    "shell_thick_nm":  (1.0, 10.0),
    "core_radius_nm":  (1.5, 8.0),
    "Nt_cm3_log10":    (16.0, 19.0),   # sample log-uniformly, 10^16 - 10^19
    "eps_shell":       (4.0, 20.0),
    "Vmax_V":          (1.0, 10.0),
}


def latin_hypercube(n_samples, n_dims, seed=42):
    """Simple LHS implementation (numpy-only, no external deps)."""
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        result[:, d] = (perm + rng.random(n_samples)) / n_samples
    return result  # values in [0,1), shape (n_samples, n_dims)


def generate_training_dataset(n_samples=5000, seed=42, cap_retention_s=1e12,
                               core_shell_pair_id=None):
    """
    Generate the full synthetic RS dataset -- this is what feeds
    Phase 4 (XGBoost) as (features -> labels) training data.

    cap_retention_s: safety ceiling only (default 1e12 s, ~32,000 years)
    to guard against pure numerical extremes for very deep traps -- NOT
    a physically-motivated cutoff like the earlier 24h cap. That 24h
    cap destroyed information: with trap depths swept up to 1.5 eV, the
    exponential Boltzmann dependence means ~79% of samples predicted
    retention beyond 24h, so nearly 4 in 5 rows collapsed to an
    identical, uninformative label. Train on log10_retention_tau_s
    (added below) instead of the raw value -- it naturally compresses
    the huge dynamic range without throwing away the signal the way a
    hard cap does.
    """
    keys = list(PARAM_RANGES.keys())
    lhs = latin_hypercube(n_samples, len(keys), seed=seed)

    rows = []
    for i in range(n_samples):
        params = {}
        for j, k in enumerate(keys):
            lo, hi = PARAM_RANGES[k]
            params[k] = lo + lhs[i, j] * (hi - lo)

        Nt_cm3 = 10 ** params["Nt_cm3_log10"]

        out = simulate_device(
            dE_LUMO_eV=params["dE_LUMO_eV"],
            dE_HOMO_eV=params["dE_HOMO_eV"],
            shell_thick_nm=params["shell_thick_nm"],
            core_radius_nm=params["core_radius_nm"],
            Nt_cm3=Nt_cm3,
            eps_shell=params["eps_shell"],
            Vmax_V=params["Vmax_V"],
        )

        retention_capped = min(out["retention_tau_s"], cap_retention_s)
        row = {
            "dE_LUMO_eV": params["dE_LUMO_eV"],
            "dE_HOMO_eV": params["dE_HOMO_eV"],
            "shell_thick_nm": params["shell_thick_nm"],
            "core_radius_nm": params["core_radius_nm"],
            "Nt_cm3": Nt_cm3,
            "eps_shell": params["eps_shell"],
            "Vmax_V": params["Vmax_V"],
            "hysteresis_window_V": out["hysteresis_window_V"],
            "on_off_ratio": out["on_off_ratio"],
            "log10_on_off_ratio": out["log10_on_off_ratio"],
            "retention_tau_s": retention_capped,
            "log10_retention_tau_s": np.log10(max(retention_capped, 1e-12)),
            "retention_was_capped": out["retention_tau_s"] > cap_retention_s,
            "core_shell_pair_id": core_shell_pair_id,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running a single test case (roughly matching the reference "
          "paper's CsPbCl3/Cs4PbCl6 confinement scale)...")
    test = simulate_device(
        dE_LUMO_eV=0.4, dE_HOMO_eV=0.4, shell_thick_nm=3.0,
        core_radius_nm=4.0, Nt_cm3=1e17, eps_shell=8.0, Vmax_V=9.0
    )
    for k, v in test.items():
        print(f"  {k}: {v}")

    print("\nGenerating synthetic training dataset (5000 samples via LHS)...")
    df = generate_training_dataset(n_samples=5000)
    df.to_csv("rs_synthetic_training_data.csv", index=False)
    print(f"Saved {len(df)} rows -> rs_synthetic_training_data.csv")
    print(df[["hysteresis_window_V", "on_off_ratio", "retention_tau_s"]].describe())