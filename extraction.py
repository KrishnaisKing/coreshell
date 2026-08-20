"""
Pull Tier-1 / Tier-2 material parameters from Materials Project (MP) and OQMD
for building a core-shell resistive-switching (RS) dataset.

Requirements (run locally, not in this sandbox):
    pip install mp-api pymatgen requests pandas

You need a free MP API key: https://next-gen.materialsproject.org/api
Set it as an environment variable: export MP_API_KEY="your_key_here"
"""

import os
import requests
import pandas as pd
from mp_api.client import MPRester

MP_API_KEY = os.environ.get("MP_API_KEY")

# Elements that anchor the halide-perovskite-style chemistry your reference
# pair (CsPbCl3-like) actually lives in. Restricting the pull to systems
# containing at least one of these keeps the pool small and relevant instead
# of screening the whole oxide-heavy MP database.
HALIDE_ANCHORS = ["Cl", "Br", "I", "F"]

def estimate_band_edges_electronegativity(formula, band_gap):
    """
    Estimate CBM/VBM vs. vacuum using the electronegativity method
    (Xu & Schoonen, Am. Mineral. 1999 / Butler-Ginley approach), commonly
    used when DFT-derived vacuum work functions are unavailable.

    chi_compound = geometric mean of constituent elements' electronegativity,
                   weighted by stoichiometric fraction.
    E_VBM_vac = -(chi_compound + 0.5*Eg)
    E_CBM_vac = -(chi_compound - 0.5*Eg)

    NOTE: this uses Pauling electronegativity (readily available via
    pymatgen) as a practical substitute for true Mulliken electronegativity
    (IE+EA)/2, which is not tabulated for all elements. This is an
    approximation, not DFT-grade -- flag accordingly in the output.
    """
    from pymatgen.core import Composition
    import numpy as np

    comp = Composition(formula)
    frac = comp.fractional_composition.as_dict()  # {element: fraction}

    log_chi_sum = 0.0
    for el_symbol, fraction in frac.items():
        from pymatgen.core.periodic_table import Element
        chi_el = Element(el_symbol).X  # Pauling electronegativity
        if chi_el is None:
            return None, None  # missing data for this element -> skip, don't fake it
        log_chi_sum += fraction * np.log(chi_el)

    chi_compound = np.exp(log_chi_sum)

    vbm_vac = -(chi_compound + 0.5 * band_gap)
    cbm_vac = -(chi_compound - 0.5 * band_gap)
    return cbm_vac, vbm_vac


def _element_count(formula):
    """Number of distinct elements in a formula. Used for the simplicity cap."""
    from pymatgen.core import Composition
    try:
        return len(Composition(formula).elements)
    except Exception:
        return 99  # unparseable -> treat as complex, gets filtered out


# -------------------------------------------------------------------
# 1. MATERIALS PROJECT — primary source (Tier 1 + most of Tier 2)
# -------------------------------------------------------------------
def fetch_mp_candidates(min_gap=0.5, max_gap=5.0, chemsys=None,
                         restrict_to_halides=True, max_elements=4):
    """
    Pull band gap and dielectric data from MP, then estimate vacuum-
    referenced CBM/VBM via the electronegativity method (Option A) since
    real DFT work-function data only exists for ~130 MP entries out of
    ~130,000+ total -- nowhere near enough coverage for a broad screen.

    restrict_to_halides: if True (default), only pull compounds containing
        at least one of Cl/Br/I/F. Your project is anchored to a
        CsPbCl3-like halide perovskite system, not oxide mineral phases,
        so this keeps the candidate pool chemically relevant instead of
        just large.
    max_elements: drop formulas with more distinct elements than this.
        Real core-shell nanoparticle materials (CsPbCl3, ZnS, CdSe, TiO2)
        are simple binaries/ternaries -- not 5+ cation mineral phases.
    """
    # NOTE: MP's `elements` filter is AND logic (material must contain ALL
    # listed elements), not OR. Passing all four halogens at once returns
    # nothing (no compound has Cl+Br+I+F simultaneously). To get "contains
    # ANY of these halides" we query each anchor element separately and
    # combine, deduplicating by material_id.
    anchor_groups = [[a] for a in HALIDE_ANCHORS] if (restrict_to_halides and chemsys is None) else [chemsys]

    seen_ids = set()
    docs = []
    with MPRester(MP_API_KEY) as mpr:
        for anchor in anchor_groups:
            batch = mpr.materials.summary.search(
                band_gap=(min_gap, max_gap),
                elements=anchor,               # None = no restriction
                fields=[
                    "material_id", "formula_pretty",
                    "band_gap", "is_gap_direct",
                    "e_electronic", "e_ionic", "e_total",  # dielectric tensors
                    "n",                            # refractive index (optional)
                    "energy_above_hull", "is_stable",  # thermodynamic stability
                ],
            )
            for d in batch:
                if d.material_id not in seen_ids:
                    seen_ids.add(d.material_id)
                    docs.append(d)

    rows = []
    for d in docs:
        if max_elements is not None and _element_count(d.formula_pretty) > max_elements:
            continue  # skip complex multi-cation phases up front

        cbm_vac, vbm_vac = estimate_band_edges_electronegativity(
            d.formula_pretty, d.band_gap
        )
        rows.append({
            "mp_id": d.material_id,
            "formula": d.formula_pretty,
            "band_gap_eV": d.band_gap,
            "is_direct_gap": d.is_gap_direct,
            "cbm_vs_vacuum_eV": cbm_vac,
            "vbm_vs_vacuum_eV": vbm_vac,
            "offset_estimation_method": "electronegativity_approx",  # transparency, not silent default
            "dielectric_electronic": _trace_avg(d.e_electronic),
            "dielectric_ionic": _trace_avg(d.e_ionic),
            "dielectric_total": _trace_avg(d.e_total),
            "refractive_index": d.n,
            "energy_above_hull_eV_per_atom": d.energy_above_hull,
            "is_stable": d.is_stable,
        })
    return pd.DataFrame(rows)


def _trace_avg(tensor):
    """Average the diagonal of a 3x3 dielectric tensor, if present."""
    if tensor is None:
        return None
    try:
        return sum(tensor[i][i] for i in range(3)) / 3.0
    except Exception:
        return None


# -------------------------------------------------------------------
# 2. OQMD — supplementary source (band gap / formation energy only)
# -------------------------------------------------------------------
def fetch_oqmd_candidates(elements=None, limit=100):
    """
    OQMD REST API (no key required): http://oqmd.org/oqmdapi/
    Returns formation energy, stability, and (sometimes) band gap.
    NOTE: OQMD does NOT provide CBM/VBM or dielectric constants —
    use this only to cross-check band gaps / expand candidate list.
    """
    base = "http://oqmd.org/oqmdapi/formationenergy"
    params = {"limit": limit, "fields": "name,entry_id,delta_e,band_gap,stability"}
    if elements:
        params["composition"] = "-".join(elements)

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("data", [])
    return pd.DataFrame(data)


# -------------------------------------------------------------------
# 3. Pairing logic — build core/shell candidate pairs
# -------------------------------------------------------------------
def _shares_real_cation(formula_core, formula_shell):
    """
    Real compatibility check: shared METAL CATION only.

    First pass excluded common anions (O, H, N, C, S) but left halogens in.
    That was a mistake once the pool itself is halide-restricted -- every
    single candidate now contains a halogen by construction, so "shared
    element" degenerates right back into a trivial pass (e.g. TlIO3 /
    InH3(IO4)2 "matching" on shared I, which isn't a real compatibility
    signal -- your reference pair shares a cation like Pb, not an anion).

    Fix: intersect on metal elements only (pymatgen Element.is_metal),
    which excludes O/H/N/C/S/halogens/other nonmetals automatically and
    actually tests for a shared cation like Cs or Pb.
    """
    from pymatgen.core import Composition
    try:
        els_core = {e for e in Composition(formula_core).elements if e.is_metal}
        els_shell = {e for e in Composition(formula_shell).elements if e.is_metal}
        return len(els_core & els_shell) > 0
    except Exception:
        return False


def build_core_shell_pairs(mp_df, core_gap_range=(2.78, 3.18), shell_gap_range=(3.20, 3.60),
                            dEc_range=(0.3, 1.5), dEv_range=(0.3, 1.5), min_gap_separation=0.3,
                            max_energy_above_hull=0.05, require_shared_element=True,
                            top_n=100, chunk_size=50):
    """
    Pair candidate 'core' materials with candidate 'shell' materials where
    shell_gap > core_gap, then compute offsets and filter to type-I in the
    SAME pass -- never materializing the full cross product in memory.

    Tighter default gap ranges (vs. the earlier 1.5-3.5 / 2.5-5.0) both
    reflect your actual target region (core ~2.98 eV, shell ~3.40 eV) and
    keep the candidate pools small enough to cross-join safely. With
    ~61,866 total MP entries, a wide-open cross join produces hundreds of
    millions of rows and blows out memory -- as you just saw.

    require_shared_element now checks for a shared CATION/halide (see
    _shares_real_cation), not just any shared element -- the old version
    passed almost everything because O is in nearly every candidate.
    """
    cores = mp_df[
        mp_df["band_gap_eV"].between(*core_gap_range) &
        mp_df["cbm_vs_vacuum_eV"].notna() & mp_df["vbm_vs_vacuum_eV"].notna() &
        (mp_df["energy_above_hull_eV_per_atom"] <= max_energy_above_hull)
    ].reset_index(drop=True)
    shells = mp_df[
        mp_df["band_gap_eV"].between(*shell_gap_range) &
        mp_df["cbm_vs_vacuum_eV"].notna() & mp_df["vbm_vs_vacuum_eV"].notna() &
        (mp_df["energy_above_hull_eV_per_atom"] <= max_energy_above_hull)
    ].reset_index(drop=True)

    # Deduplicate: MP has many polymorphs per composition (different mp_ids,
    # same formula). Keep only the most stable entry per unique formula --
    # having 5 different structures of the same compound isn't 5 real
    # candidates, it's 1 candidate counted 5 times.
    cores = cores.sort_values("energy_above_hull_eV_per_atom").drop_duplicates(
        subset="formula", keep="first"
    ).reset_index(drop=True)
    shells = shells.sort_values("energy_above_hull_eV_per_atom").drop_duplicates(
        subset="formula", keep="first"
    ).reset_index(drop=True)

    max_possible = len(cores) * len(shells)
    per_chunk = len(cores) * chunk_size
    print(f"Candidate cores: {len(cores)}, candidate shells: {len(shells)} "
          f"-> max possible pairs: {max_possible:,} (never fully materialized; "
          f"processed as {len(shells)//chunk_size + 1} chunks of ~{per_chunk:,} rows each)")

    if per_chunk > 20_000_000:
        raise ValueError(
            f"Per-chunk size ({per_chunk:,} rows) is too large for safe memory use. "
            f"Reduce chunk_size (currently {chunk_size}) or narrow core_gap_range."
        )

    results = []
    core_cols = ["mp_id", "formula", "band_gap_eV", "cbm_vs_vacuum_eV", "vbm_vs_vacuum_eV"]
    shell_cols = core_cols

    # Process in chunks over shells to bound peak memory regardless of pool size
    n_chunks = len(shells) // chunk_size + 1
    total_matches = 0
    for i, start in enumerate(range(0, len(shells), chunk_size)):
        if i % 20 == 0:
            print(f"  chunk {i+1}/{n_chunks} ({total_matches} matching rows so far)...")
        shell_chunk = shells.iloc[start:start + chunk_size]

        cores["key"] = 1
        shell_chunk = shell_chunk.copy()
        shell_chunk["key"] = 1

        merged = cores[core_cols + ["key"]].merge(
            shell_chunk[shell_cols + ["key"]], on="key", suffixes=("_core", "_shell")
        ).drop("key", axis=1)

        merged = merged[merged["band_gap_eV_shell"] > merged["band_gap_eV_core"]]
        # Require a real gap separation, not just any positive difference -
        # without this, dEc+dEv is often trivially small and both offsets
        # clear a low 0.1 eV floor almost by default (this is what caused
        # the 5.3M-row blowup previously).
        merged = merged[
            (merged["band_gap_eV_shell"] - merged["band_gap_eV_core"]) >= min_gap_separation
        ]

        merged["dEc_eV"] = merged["cbm_vs_vacuum_eV_shell"] - merged["cbm_vs_vacuum_eV_core"]
        merged["dEv_eV"] = merged["vbm_vs_vacuum_eV_core"] - merged["vbm_vs_vacuum_eV_shell"]
        merged["is_type1_shell_confines_core"] = (
            (merged["dEc_eV"] > 0) & (merged["dEv_eV"] > 0)
        )

        # Keep only rows that are actually type-I AND in your target sweep
        # range -- filtering here, not after, is what keeps memory bounded.
        keep = merged[
            merged["is_type1_shell_confines_core"] &
            merged["dEc_eV"].between(*dEc_range) &
            merged["dEv_eV"].between(*dEv_range)
        ].copy()

        if require_shared_element and len(keep):
            keep = keep[keep.apply(
                lambda row: _shares_real_cation(row["formula_core"], row["formula_shell"]),
                axis=1
            )]

        if len(keep):
            results.append(keep)
            total_matches += len(keep)

    print(f"  chunk {n_chunks}/{n_chunks} ({total_matches} matching rows so far)... done")

    if not results:
        return pd.DataFrame()
    all_pairs = pd.concat(results, ignore_index=True)

    # Rank by confinement strength (dEc+dEv) rather than returning
    # everything that merely clears the thresholds -- this is what turns
    # "94,000 numeric matches" into an actual shortlist to inspect.
    all_pairs["confinement_score_eV"] = all_pairs["dEc_eV"] + all_pairs["dEv_eV"]
    all_pairs = all_pairs.sort_values("confinement_score_eV", ascending=False)

    return all_pairs.head(top_n).reset_index(drop=True) if top_n else all_pairs


def enforce_type1_offsets(pairs_df, dEc_range=(0.1, 1.5), dEv_range=(0.1, 1.5)):
    """
    After you've computed dEc_eV and dEv_eV using a common vacuum
    reference (see note above), keep only physically valid type-I
    pairs: both offsets positive (shell confines core) and within
    your target sweep range.
    """
    df = pairs_df.copy()
    mask = (
        df["dEc_eV"].between(*dEc_range) &
        df["dEv_eV"].between(*dEv_range)
    )
    return df[mask]


if __name__ == "__main__":
    if not MP_API_KEY:
        raise SystemExit("Set MP_API_KEY environment variable first.")

    import os as _os
    cache_path = "mp_candidates_raw.csv"
    if _os.path.exists(cache_path):
        print(f"Using cached {cache_path} (delete this file to refetch from MP)")
        mp_df = pd.read_csv(cache_path)
    else:
        mp_df = fetch_mp_candidates(chemsys=None, restrict_to_halides=True, max_elements=4)
        if mp_df.empty:
            raise SystemExit("MP query returned 0 candidates -- check MP_API_KEY and query filters before continuing.")
        mp_df.to_csv(cache_path, index=False)
        print(f"Pulled {len(mp_df)} MP candidates (halide-restricted, <=4 elements) -> {cache_path}")

    try:
        oqmd_df = fetch_oqmd_candidates(limit=200)
        oqmd_df.to_csv("oqmd_candidates_raw.csv", index=False)
        print(f"Pulled {len(oqmd_df)} OQMD candidates -> oqmd_candidates_raw.csv")
    except Exception as e:
        print(f"OQMD fetch failed ({e}) — skipping, not required for core-shell pairing.")

    pairs = build_core_shell_pairs(mp_df)
    pairs.to_csv("core_shell_pairs_type1_in_range.csv", index=False)
    print(f"{len(pairs)} pairs are type-I (shell confines core) AND within "
          f"the dEc/dEv sweep range used in build_core_shell_pairs() -> core_shell_pairs_type1_in_range.csv")