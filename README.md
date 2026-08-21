# Core-Shell Type-I Heterostructure Resistive Switching — ML Prediction Pipeline

## Objective

Predict resistive-switching (RS) behavior — hysteresis window, ON/OFF ratio, and
retention time — for type-I band-aligned core-shell nanoparticles, using a
physics-informed simulator to generate training labels and XGBoost to learn the
structure-property relationship. Reference system: CsPbCl₃ (core) / Cs₄PbCl₆
(shell), per Bera et al., *Appl. Phys. Lett.* 119, 223501 (2021).

**Novelty claim:** ML as a fast, invertible surrogate for band-alignment-driven
RS in a mechanism (type-I confinement in core-shell nanoparticles) that lacks an
established compact model — not "ML predicts RS metrics" in general, which is
already standard in the memristor-modeling literature.

---

## Pipeline Overview

```
Phase 1: Data Extraction (MP + OQMD)
Phase 2: Type-I Heterostructure Filtering
Phase 3: Physics-Informed Feature Engineering
Phase 4: Physics Simulator (labels) + K-Means/XGBoost (training)
Phase 5: Lattice Strain & Synthesizability Index
```

### Phase 1 — Data Extraction (`fetch_mp_oqmd.py`)
Pulls band gap, dielectric tensors, energy above hull, and formula/composition
for candidate materials from Materials Project (primary source). OQMD attempted
as a secondary source but contributes minimally — no CBM/VBM or dielectric data,
and its public API is unreliable (frequent timeouts). Treat OQMD as best-effort,
not required.

CBM/VBM are estimated via the **electronegativity method** (Xu & Schoonen 1999 /
Butler-Ginley), using Pauling electronegativity as a practical substitute for
true Mulliken electronegativity, since real DFT work-function data exists for
only ~130 of ~130,000+ MP entries — far too sparse for a broad screen. Every row
is tagged `offset_estimation_method` for traceability.

**Known accuracy issue:** this method systematically **overestimates band gaps
by ~1.4 eV for Bi- and Tl-containing compounds** (validated against literature
for Cs₂KInCl₆ — excellent agreement, 3.578 eV predicted vs. 3.6–3.72 eV
reported — versus Cs₂RbBiBr₆ and Cs₂ScTlCl₆, both substantially overestimated).
Candidates containing Bi/Tl should be flagged or excluded until corrected via
DFT or literature lookup.

### Phase 2 — Type-I Heterostructure Filtering (`fetch_mp_oqmd.py`, `build_core_shell_pairs`)
Screens core-shell pairs for genuine type-I alignment (both ΔEc, ΔEv > 0,
computed on a common vacuum-referenced scale — not raw MP CBM/VBM, which are
Fermi-referenced per-material and not directly comparable across compounds).

Filters applied, in order:
- Gap window matching target region (core ~2.98 eV, shell ~3.40 eV)
- Thermodynamic stability (energy above hull ≤ 0.05 eV/atom)
- Deduplication of polymorphs (keep most stable structure per formula)
- Shared metal cation requirement (real chemical compatibility check — not
  shared *any* element, which trivially passes for oxygen-containing pairs)
- Minimum offset magnitude (dEc, dEv ≥ 0.3 eV) and minimum gap separation
  (≥0.3 eV) to avoid numerically-trivial "passes"
- Ranked by confinement score (dEc + dEv), top-100 shortlist retained

**Result:** 100 candidate pairs, 56 unique cores / 51 unique shells — no single
material dominates. Top candidates are halide double perovskites (Cs₂KInCl₆,
Cs₂RbBiBr₆, Cs₂ScTlCl₆, etc.), a real, literature-relevant materials family.

### Phase 3 — Feature Engineering
Derived descriptors: band offsets (dEc, dEv), confinement score, dielectric
contrast (where DFPT data available — rare, see Phase 1), barrier asymmetry.
**"Defect potential gradient" is not derivable from bulk MP data** (requires
dedicated defect-formation-energy calculations, not available at scale) — do
not claim this as a real extracted feature; trap density is a *simulator sweep
variable*, not a materials-database feature.

### Phase 4 — Physics Simulator + ML Training

**Simulator (`rs_simulator.py`):** generates RS labels via Simmons tunneling
(with a Fowler-Nordheim high-field branch for cases where qV exceeds the
barrier height — the dominant regime given 0.1–1.5 eV barriers and few-volt
biases) and Poole-Frenkel trap-release kinetics.

Calibrated against the reference paper's anchor points:
- 10 ms write pulse width (used directly, not an arbitrary value)
- Retention: 2192 s predicted vs. 2400 s (40 min) reported — **~9% agreement**
  at the calibration reference point (0.3 eV trap depth)
- Switching voltage threshold (~8 V in the paper) — **not reproduced**; at
  these barrier/thickness scales the device sits in field-emission for
  essentially any voltage above ~0.5 V, so no clean threshold emerges from
  this simplified model. Documented limitation, not silently ignored.

**Every parameter-output relationship was individually validated**, not just
aggregate-correlated (aggregate correlation across a 7-parameter random sweep
can mask real single-variable effects — several were confirmed via isolated
single-parameter sweeps rather than trusted from correlation alone):

| Parameter | Output | Relationship | Verified via |
|---|---|---|---|
| dE_LUMO (trap depth) | Retention | Strong, 0.90 (log-scale) | Aggregate + calibration point |
| Shell thickness | ON/OFF, hysteresis | Real, monotonic | Isolated sweep (aggregate correlation was misleadingly low) |
| N_t (trap density) | ON/OFF, hysteresis | Real but concentrated at high density + large core | Isolated sweep |
| N_t | Retention | Correctly near-zero | Physically expected (release rate is a per-trap property) |

**K-Means + XGBoost (`kmeans_xgboost_train.py`):** clusters the 44 real
candidates (that passed Phase 5's lattice filter) by material-level descriptors
into 8 groups, assigns *whole clusters* to train/test (never splitting a single
candidate's device-sweep rows across both) — verified via a hard assertion, not
just a stated intention. This directly avoids the leakage pattern found in a
prior version of this project (see **Known Limitations** below).

Results (group-safe split, 33 train / 11 test candidates, 1650/550 rows):

| Target | R² | MAE |
|---|---|---|
| Hysteresis window (V) | 0.954 | 0.045 |
| log10(ON/OFF ratio) | 0.945 | 0.020 |
| log10(retention, s) | 0.971 | 0.116 |

### Phase 5 — Lattice Strain & Synthesizability (`phase5_lattice_mismatch.py`)
Computes pseudo-cubic lattice parameter `a_pc = (V_cell/Z)^(1/3)` for each
candidate from real MP structure data, and interfacial mismatch
`η = (a_shell − a_core)/a_core`. Standard ±6% rule-of-thumb applied for
"coherent epitaxial" classification. 44/100 Phase 2 candidates pass this filter
and feed into Phase 4's real-candidate training set.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install mp-api pymatgen requests pandas numpy scipy scikit-learn xgboost matplotlib
export MP_API_KEY="your_key_here"  # get free key at next-gen.materialsproject.org
```

## Run order

```bash
python fetch_mp_oqmd.py                  # Phase 1-2 -> core_shell_pairs_type1_in_range.csv
python phase5_lattice_mismatch.py        # Phase 5 -> core_shell_pairs_with_lattice.csv
python mp_phase4_real_candidates.py      # hooks real candidates into rs_simulator.py
python kmeans_xgboost_train.py           # Phase 4 -> trained models + predictions_*.csv
python parity_plots.py                   # visual sanity check
```

---

## Known Limitations (read before presenting results)

1. **Test set is 11 materials, not "n=550."** The 550 test rows come from only
   11 unique candidates (50 device-sweep points each) — row count overstates
   the real statistical power of the generalization test. Results have not
   been re-run across multiple K-Means seeds/splits to check stability.

2. **Retention R²=0.971 likely reduces mostly to a per-material lookup.**
   `dE_LUMO_eV` is fixed per candidate and already correlates 0.90 with
   log-retention on its own; device-sweep parameters contribute comparatively
   little. No baseline (e.g., predict the training-set mean per material
   cluster) has been run to confirm XGBoost adds real value beyond this.

3. **All labels are simulator-generated, calibrated against a single anchor
   point** (0.3 eV trap depth → retention, ~9% agreement). The simulator's
   other physics (bipolar reset behavior, ~5-8% run-to-run variance reported
   in the paper, pulse-width independence) has not been separately validated.
   A high R² here demonstrates XGBoost reproduces the simulator's internal
   rules — it does not by itself demonstrate agreement with real device
   physics beyond that one calibration point.

4. **Voltage threshold (~8V in the paper) is not reproduced.** Documented and
   accepted as a simplification (Option B decision) — the model treats
   voltage as a smooth, monotonic capture-efficiency driver rather than a
   sharp switching threshold.

5. **Bi/Tl-containing candidates have known-overestimated band gaps** (~1.4 eV
   high) from the Phase 1 electronegativity approximation. These candidates'
   downstream simulator outputs, and any model trained on them, inherit this
   error until corrected via DFT or literature values.

6. **Hysteresis and ON/OFF ratio are substantially device-property-driven**
   (voltage, shell thickness), not purely material-property-driven. The
   materials-screening pipeline's predictive value is strongest for
   **retention** (material-dominated); hysteresis/ON-OFF ratio could plausibly
   be tuned similarly across different materials just by varying device
   geometry and operating voltage. State this explicitly rather than implying
   material screening is equally important for all three outputs.

7. **A prior version of this project's ML stage showed clear data leakage**
   (near-perfect 1:1 parity with a hard ceiling artifact, single-feature
   dominance in importance rankings) traced to the training label being
   algebraically derived from the same features used to predict it. The
   current pipeline's group-safe K-Means split and simulator-derived
   (feature-independent) labels are the direct fix for this — mentioned here
   so the distinction is clear if compared against earlier project output.

---

## File Reference

| File | Purpose |
|---|---|
| `fetch_mp_oqmd.py` | Phase 1-2: MP/OQMD extraction, type-I filtering, candidate ranking |
| `phase5_lattice_mismatch.py` | Phase 5: structure fetch, lattice mismatch, synthesizability ranking |
| `rs_simulator.py` | Physics simulator: Simmons + FN tunneling, Poole-Frenkel trap kinetics |
| `calibrate.py` | Calibration check against paper's anchor points |
| `mp_phase4_real_candidates.py` | Hooks real candidate dEc/dEv/lattice data into simulator sweeps |
| `kmeans_xgboost_train.py` | Group-safe train/test split + XGBoost training |
| `parity_plots.py` | Visual model-quality sanity check |

## Suggested Next Steps

- Re-run K-Means/XGBoost across multiple random seeds to check R² stability
- Add a naive per-cluster-mean baseline to quantify XGBoost's real marginal value
- Correct or exclude Bi/Tl-containing candidates pending better band-gap estimates
- Validate simulator against additional paper anchor points beyond retention
  (bipolar reset, run-to-run variance)
- Consider inverse design (target RS metrics → recommend material/geometry) as
  the paper's most clearly novel contribution, per the original novelty framing