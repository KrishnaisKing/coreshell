"""
Phase 4 (real candidates) -- run the physics simulator (simulate_device,
from rs_simulator.py) against the ACTUAL screened core/shell candidates
that passed both Phase 3 (type-I confinement) and Phase 5 (lattice
mismatch), instead of purely synthetic swept parameter combinations.

This is what turns the earlier purely-synthetic rs_synthetic_training_data.csv
into a training set actually tied to real MP-screened materials: dE_LUMO_eV
and dE_HOMO_eV are no longer swept -- they're fixed per candidate to the
real dEc_eV/dEv_eV computed from MP band-edge data. eps_shell uses the
real DFT dielectric constant when MP has one for the shell material.

What's still swept: shell_thick_nm, core_radius_nm, Nt_cm3, Vmax_V. These
are DEVICE/SYNTHESIS choices (how thick you grow the shell, what particle
size you target, doping/defect density, drive voltage) -- not properties
of the material itself, so there's no "real" value to look up for them.
Each candidate gets its own small LHS sweep over these four so the model
still sees how device geometry affects a given material pair.

Run locally:
    pip install pandas numpy
    python mp_phase4_real_candidates.py

Requires, in the same folder:
    core_shell_pairs_with_lattice.csv  (Phase 5 output)
    mp_candidates_raw.csv              (Phase 1 output -- for eps_shell lookup)
    rs_simulator.py                    (the simulate_device physics module --
                                         rename your simulator file to this,
                                         or edit the import line below)
"""

import os
import numpy as np
import pandas as pd

from rs_simulator import simulate_device, latin_hypercube

# Free (still-swept) device/geometry parameters -- material-fixed params
# (dE_LUMO_eV, dE_HOMO_eV, eps_shell) come from the real candidate data
# instead, see build_dataset() below.
FREE_PARAM_RANGES = {
    "shell_thick_nm":  (1.0, 10.0),
    "core_radius_nm":  (1.5, 8.0),
    "Nt_cm3_log10":    (16.0, 19.0),   # log-uniform, matches original sweep
    "Vmax_V":          (1.0, 10.0),
}

DEFAULT_EPS_SHELL_FALLBACK_RANGE = (4.0, 20.0)  # used only when MP has no
                                                  # DFT dielectric constant
                                                  # for a given shell material


def load_eps_shell_lookup(mp_candidates_path="mp_candidates_raw.csv"):
    """
    Build {mp_id: dielectric_total} from the Phase 1 MP pull, so real
    candidates can use their actual computed dielectric constant instead
    of a swept placeholder. MP's DFPT dielectric coverage is incomplete
    (not every material has it computed) -- entries with no value are
    simply absent from this dict, and the caller falls back explicitly
    rather than silently defaulting to something arbitrary.
    """
    df = pd.read_csv(mp_candidates_path)
    df = df.dropna(subset=["dielectric_total"])
    return dict(zip(df["mp_id"], df["dielectric_total"]))


def build_dataset(pairs_path="core_shell_pairs_with_lattice.csv",
                   mp_candidates_path="mp_candidates_raw.csv",
                   n_points_per_candidate=50,
                   only_lattice_ok=True,
                   cap_retention_s=1e12,
                   seed=42):
    pairs_df = pd.read_csv(pairs_path)
    if only_lattice_ok:
        n_before = len(pairs_df)
        pairs_df = pairs_df[pairs_df["lattice_mismatch_ok"] == True].reset_index(drop=True)
        print(f"Using {len(pairs_df)}/{n_before} candidates that passed the lattice mismatch filter.")

    eps_lookup = load_eps_shell_lookup(mp_candidates_path)

    keys = list(FREE_PARAM_RANGES.keys())
    rng = np.random.default_rng(seed)

    rows = []
    for _, cand in pairs_df.iterrows():
        pair_id = f"{cand['mp_id_core']}__{cand['mp_id_shell']}"
        dE_LUMO_eV = cand["dEc_eV"]   # real MP-derived conduction-band offset
        dE_HOMO_eV = cand["dEv_eV"]   # real MP-derived valence-band offset

        shell_mp_id = cand["mp_id_shell"]
        eps_from_dft = eps_lookup.get(shell_mp_id)
        if eps_from_dft is not None and eps_from_dft > 0:
            eps_shell_val = eps_from_dft
            eps_source = "dft"
        else:
            # per-candidate fixed fallback draw (not re-swept every point --
            # keeps eps_shell consistent within a candidate's own sweep,
            # same as the real DFT case would be)
            lo, hi = DEFAULT_EPS_SHELL_FALLBACK_RANGE
            eps_shell_val = lo + rng.random() * (hi - lo)
            eps_source = "swept_fallback"

        lhs = latin_hypercube(n_points_per_candidate, len(keys), seed=hash(pair_id) % (2**31))

        for i in range(n_points_per_candidate):
            params = {}
            for j, k in enumerate(keys):
                lo, hi = FREE_PARAM_RANGES[k]
                params[k] = lo + lhs[i, j] * (hi - lo)
            Nt_cm3 = 10 ** params["Nt_cm3_log10"]

            out = simulate_device(
                dE_LUMO_eV=dE_LUMO_eV,
                dE_HOMO_eV=dE_HOMO_eV,
                shell_thick_nm=params["shell_thick_nm"],
                core_radius_nm=params["core_radius_nm"],
                Nt_cm3=Nt_cm3,
                eps_shell=eps_shell_val,
                Vmax_V=params["Vmax_V"],
            )

            retention_capped = min(out["retention_tau_s"], cap_retention_s)
            rows.append({
                "core_shell_pair_id": pair_id,
                "mp_id_core": cand["mp_id_core"],
                "formula_core": cand["formula_core"],
                "mp_id_shell": cand["mp_id_shell"],
                "formula_shell": cand["formula_shell"],
                "lattice_mismatch_pct": cand["lattice_mismatch_pct"],
                "confinement_score_eV": cand["confinement_score_eV"],
                "dE_LUMO_eV": dE_LUMO_eV,
                "dE_HOMO_eV": dE_HOMO_eV,
                "eps_shell": eps_shell_val,
                "eps_shell_source": eps_source,
                "shell_thick_nm": params["shell_thick_nm"],
                "core_radius_nm": params["core_radius_nm"],
                "Nt_cm3": Nt_cm3,
                "Vmax_V": params["Vmax_V"],
                "hysteresis_window_V": out["hysteresis_window_V"],
                "on_off_ratio": out["on_off_ratio"],
                "log10_on_off_ratio": out["log10_on_off_ratio"],
                "retention_tau_s": retention_capped,
                "log10_retention_tau_s": np.log10(max(retention_capped, 1e-12)),
                "retention_was_capped": out["retention_tau_s"] > cap_retention_s,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("rs_training_data_real_candidates.csv", index=False)
    print(f"Saved {len(df)} rows ({df['core_shell_pair_id'].nunique()} unique candidates "
          f"x up to 50 device-param sweep points each) -> rs_training_data_real_candidates.csv")
    n_dft = (df["eps_shell_source"] == "dft").sum()
    print(f"eps_shell from real DFT dielectric: {n_dft}/{len(df)} rows "
          f"({df.loc[df['eps_shell_source']=='dft','core_shell_pair_id'].nunique()} candidates)")
    print(df[["hysteresis_window_V", "on_off_ratio", "retention_tau_s"]].describe())