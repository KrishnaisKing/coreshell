"""
Phase 5 -- lattice mismatch screening for the core/shell candidate pairs
produced by mp_core_shell_screen.py.

Run locally (not in this sandbox -- Materials Project needs your MP_API_KEY
and this environment has no network access to materialsproject.org):
    pip install mp-api pymatgen pandas
    export MP_API_KEY="your_key_here"
    python mp_lattice_mismatch.py

Why this matters for the pipeline: dEc/dEv screening (Phase 3) tells you
whether a shell electronically confines a core. It says nothing about
whether the shell can actually GROW on the core without so much strain
that the interface fills with misfit dislocations / traps, which would
blow up the same defect-trap physics the simulator (Phase 4) is modeling.
A type-I pair with huge lattice mismatch is a candidate that looks good
electronically and is unrealistic to actually synthesize cleanly.

Method: PSEUDO-CUBIC LATTICE PARAMETER
Most core/shell candidates here are not simple cubic perovskites -- they're
halide double perovskites, layered halides, etc. There's no single "a"
to compare directly. The standard workaround used in the halide-perovskite
literature (Kieslich et al., 2014 and follow-ons) is the PSEUDO-CUBIC
lattice constant:
    a_pc = (V_cell / Z) ** (1/3)
where V_cell is the DFT relaxed unit cell volume and Z is the number of
formula units per cell (so V_cell/Z is the volume "per formula unit",
treated as if it were a cube). This lets you compare a rock-salt-derived
double perovskite against a simple perovskite on equal footing.

CAVEAT (stated plainly, not hidden): pseudo-cubic a is an approximation.
For structures that are very anisotropic (layered, low-symmetry) it can
understate the real mismatch along specific crystallographic directions.
Treat the output as a first-pass screen, not a substitute for actually
checking the interface geometry of your top few finalists by hand.
"""

import os
import time
import pandas as pd
import numpy as np
from mp_api.client import MPRester

MP_API_KEY = os.environ.get("MP_API_KEY")

# Default filter: pairs are kept if |mismatch| <= this. 5-8% is the
# rough "commonly synthesizable with coherent/semi-coherent interface"
# range cited across core/shell nanocrystal literature (CdSe/ZnS-style
# systems tolerate up to ~10-12% via strain relaxation at the nanoscale,
# but that's the exception, not the default expectation) -- set generously
# at 10% so nothing borderline is silently dropped before you look at it.
DEFAULT_MAX_MISMATCH_PCT = 10.0


def _pseudocubic_a(structure):
    """
    a_pc = (V_cell / Z) ** (1/3), Z = formula units per cell.
    Returns None if Z can't be determined (shouldn't happen for a
    well-formed MP structure, but don't silently divide by a bad Z).
    """
    try:
        _, Z = structure.composition.get_reduced_composition_and_factor()
        if Z is None or Z <= 0:
            return None
        return (structure.volume / Z) ** (1.0 / 3.0)
    except Exception:
        return None


def fetch_lattice_params(mp_ids, chunk_size=100, pause_s=0.2):
    """
    Batch-fetch structures for a list of MP IDs and compute pseudo-cubic
    a (Angstrom) for each. Chunked because MP's API has practical limits
    on how many material_ids you can request in a single call, and a
    small pause between chunks is polite to the API (not required, just
    avoids hammering it on a large candidate list).
    """
    unique_ids = sorted(set(mp_ids))
    results = {}
    with MPRester(MP_API_KEY) as mpr:
        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i:i + chunk_size]
            docs = mpr.materials.summary.search(
                material_ids=chunk,
                fields=["material_id", "structure"],
            )
            for d in docs:
                a_pc = _pseudocubic_a(d.structure)
                results[str(d.material_id)] = a_pc
            if i + chunk_size < len(unique_ids):
                time.sleep(pause_s)
    return results  # {mp_id: a_pc_angstrom or None}


def add_lattice_mismatch(pairs_df, max_mismatch_pct=DEFAULT_MAX_MISMATCH_PCT):
    """
    Takes the Phase-3 shortlist (core_shell_pairs_type1_in_range.csv,
    columns mp_id_core / mp_id_shell already present) and adds:
        a_pc_core_ang, a_pc_shell_ang   -- pseudo-cubic lattice constants
        lattice_mismatch_pct             -- (a_shell - a_core)/a_core * 100
        lattice_mismatch_ok              -- |mismatch| <= max_mismatch_pct
    Rows where either structure's a_pc couldn't be computed keep
    lattice_mismatch_pct = NaN and lattice_mismatch_ok = False -- they are
    NOT silently dropped, just flagged, since "couldn't compute" and
    "computed and bad" are different things and collapsing them would
    hide which rows need a manual look.
    """
    all_ids = list(pairs_df["mp_id_core"]) + list(pairs_df["mp_id_shell"])
    print(f"Fetching structures for {len(set(all_ids))} unique MP IDs...")
    a_pc_map = fetch_lattice_params(all_ids)

    n_missing = sum(1 for v in a_pc_map.values() if v is None)
    if n_missing:
        print(f"  warning: {n_missing} structures returned no usable pseudo-cubic a "
              f"(missing structure data or degenerate cell) -- affected pairs will be flagged, not dropped.")

    df = pairs_df.copy()
    df["a_pc_core_ang"] = df["mp_id_core"].map(a_pc_map)
    df["a_pc_shell_ang"] = df["mp_id_shell"].map(a_pc_map)

    valid = df["a_pc_core_ang"].notna() & df["a_pc_shell_ang"].notna()
    df["lattice_mismatch_pct"] = np.nan
    df.loc[valid, "lattice_mismatch_pct"] = (
        (df.loc[valid, "a_pc_shell_ang"] - df.loc[valid, "a_pc_core_ang"])
        / df.loc[valid, "a_pc_core_ang"] * 100.0
    )

    df["lattice_mismatch_ok"] = False
    df.loc[valid, "lattice_mismatch_ok"] = df.loc[valid, "lattice_mismatch_pct"].abs() <= max_mismatch_pct

    return df


if __name__ == "__main__":
    if not MP_API_KEY:
        raise SystemExit("Set MP_API_KEY environment variable first.")

    in_path = "core_shell_pairs_type1_in_range.csv"
    if not os.path.exists(in_path):
        raise SystemExit(
            f"{in_path} not found -- run mp_core_shell_screen.py first to produce "
            f"the Phase-3 shortlist this script reads."
        )

    pairs_df = pd.read_csv(in_path)
    out_df = add_lattice_mismatch(pairs_df)
    out_df.to_csv("core_shell_pairs_with_lattice.csv", index=False)

    n_ok = out_df["lattice_mismatch_ok"].sum()
    n_total = len(out_df)
    print(f"{n_ok}/{n_total} pairs within +/-{DEFAULT_MAX_MISMATCH_PCT}% pseudo-cubic lattice "
          f"mismatch -> core_shell_pairs_with_lattice.csv")
    print(out_df.sort_values("lattice_mismatch_pct", key=lambda s: s.abs())
                [["formula_core", "formula_shell", "a_pc_core_ang", "a_pc_shell_ang",
                  "lattice_mismatch_pct", "lattice_mismatch_ok", "confinement_score_eV"]]
                .head(15))