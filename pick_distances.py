#!/usr/bin/env python3
"""Distance from every CryoSPARC pick to the nearest simulated (ground-truth) particle.

Reads
  * a CryoSPARC particle .cs file (numpy structured array; needs location/* fields), and
  * the TEM-simulator ground-truth coord_<micid>_<species>.star files (searched recursively),
and reports, per pick, the distance (in Angstrom) to the nearest ground-truth particle, its
species and Euler angles, and the distance to the nearest particle of any OTHER species. It then
tabulates how the labelling radius (a fraction of the micrograph width, as in training_data.py)
changes the species counts, and how many picks are ambiguous (a particle of another species is
closer than --ambiguity-radius-A). It writes labels.csv with one label per pick: the species,
'Empty' (no particle within --radius), 'Ambiguous' (dropped), or 'NoGT' (no star file found for
that micrograph at all).

The actual per-pick labelling rule lives in labeling.py (label_picks), shared with
training_data.py so the two can't drift apart; this script is just a CryoSPARC-free offline
diagnostic/validation harness around it.

Optional: --refine-cs <cs with alignments3D/shift> joins by uid and plots |refined shift| against
the pick-to-nearest-particle distance (sign-agnostic: magnitudes only).

    python pick_distances.py --cs /mnt/data/CS-lonp/J58/extracted_particles.cs \
        --star-root /mnt/data/lonp --out pick_distances_out

Dependencies: numpy, pandas, scipy; matplotlib for the plots (skipped if missing).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import labeling


def read_star(path: Path) -> pd.DataFrame:
    """Minimal reader for single-loop RELION star files (no `starfile` dependency)."""
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if t == "loop_":
                in_loop = True
            elif in_loop and t.startswith("_rln"):
                cols.append(t.split()[0].lstrip("_"))
            elif in_loop and cols and not t.startswith(("data_", "_")):
                vals = t.split()
                if len(vals) == len(cols):  # silently drops truncated rows
                    rows.append(vals)
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c != "rlnStructure":
            df[c] = df[c].astype(float)
    return df


def index_star_files(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("coord_*_*.star"):
        idx[p.name.split("_")[1]].append(p)
    return idx


def decode(a) -> str:
    return a.decode() if isinstance(a, bytes) else str(a)


def load_job(cs_path: Path, star_root: Path, mic_dim: int = 4096, max_picks: int | None = None):
    """Load a CryoSPARC particle .cs file plus its matching ground-truth star files.

    Returns (picks_df, mic_ids, a_per_frac, star_idx):
      - picks_df: the raw .cs structured array wrapped minimally (just what callers need: uid,
        center_x_frac, center_y_frac -- kept as the original numpy structured array for simplicity)
      - mic_ids: (n,) array of micrograph ids, one per pick
      - a_per_frac: (n,) Angstrom per unit fractional coordinate, one per pick
      - star_idx: dict mic_id -> list of star file paths (from index_star_files)
    """
    cs = np.load(cs_path, allow_pickle=False)
    if max_picks and len(cs) > max_picks:
        cs = cs[np.random.default_rng(0).choice(len(cs), max_picks, replace=False)]

    a_per_frac = labeling.angstrom_per_frac(cs["location/micrograph_shape"], cs["location/micrograph_psize_A"])
    mic_ids = np.array([Path(decode(p)).stem.rsplit("_", 1)[-1] for p in cs["location/micrograph_path"]])
    star_idx = index_star_files(star_root)
    return cs, mic_ids, a_per_frac, star_idx


def label_all(
    picks_df,
    mic_ids: np.ndarray,
    a_per_frac: np.ndarray,
    star_idx: dict[str, list[Path]],
    match_radius_A: float,
    ambiguity_radius_A: float,
    mic_dim: int = 4096,
) -> pd.DataFrame:
    """Label every pick in picks_df against its micrograph's ground truth, via labeling.label_picks."""
    cs = picks_df
    n = len(cs)
    pick_frac = np.stack([cs["location/center_x_frac"], cs["location/center_y_frac"]], axis=1).astype(float)

    out = {
        "label": np.full(n, labeling.NO_GT, dtype=object),
        "species_nearest": np.full(n, labeling.NO_GT, dtype=object),
        "dist_A": np.full(n, np.nan), "dist_other_A": np.full(n, np.nan),
        "rot": np.full(n, np.nan), "tilt": np.full(n, np.nan), "psi": np.full(n, np.nan),
        "dx_A": np.full(n, np.nan), "dy_A": np.full(n, np.nan),
    }

    for mic in np.unique(mic_ids):
        sel = np.where(mic_ids == mic)[0]
        paths = star_idx.get(mic)
        gt = None
        if paths:
            star_df = pd.concat([read_star(p) for p in paths], ignore_index=True)
            gt = labeling.GroundTruth.from_coords(
                star_df["rlnCoordinateX"].to_numpy(), star_df["rlnCoordinateY"].to_numpy(),
                star_df["rlnStructure"].to_numpy(),
                star_df["rlnAngleRot"].to_numpy(), star_df["rlnAngleTilt"].to_numpy(), star_df["rlnAnglePsi"].to_numpy(),
                mic_dim,
            )

        result = labeling.label_picks(
            pick_frac[sel, 0], pick_frac[sel, 1], gt, a_per_frac=a_per_frac[sel[0]],
            match_radius_A=match_radius_A, ambiguity_radius_A=ambiguity_radius_A,
        )
        out["label"][sel] = result.label
        out["species_nearest"][sel] = result.species_nearest
        out["dist_A"][sel] = result.dist_A
        out["dist_other_A"][sel] = result.dist_other_A
        out["rot"][sel] = result.rot
        out["tilt"][sel] = result.tilt
        out["psi"][sel] = result.psi
        out["dx_A"][sel] = result.dx_A
        out["dy_A"][sel] = result.dy_A

    return pd.DataFrame({"uid": cs["uid"], "mic_id": mic_ids, **out})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", required=True, type=Path)
    ap.add_argument("--star-root", required=True, type=Path)
    ap.add_argument("--mic-dim", type=int, default=4096, help="simulated micrograph size in px (as in training_data.py)")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05],
                    help="labelling radii as fractions of the micrograph width, for the sweep table")
    ap.add_argument("--radius", type=float, default=0.01, help="labelling radius (fraction of micrograph width) for labels.csv")
    ap.add_argument("--ambiguity-radius-A", type=float, default=None,
                    help="a pick is ambiguous if a particle of another species is closer than this (A); default: the labelling radius in A")
    ap.add_argument("--out", type=Path, default=Path("pick_distances_out"))
    ap.add_argument("--refine-cs", type=Path, default=None)
    ap.add_argument("--max-picks", type=int, default=None, help="subsample for a quick test")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cs, mic_ids, a_per_frac, star_idx = load_job(args.cs, args.star_root, mic_dim=args.mic_dim, max_picks=args.max_picks)
    print(f"{len(cs)} picks in {args.cs}")
    print(f"micrograph at {a_per_frac[0] / args.mic_dim:.4f} A/px -> {a_per_frac[0]:.1f} A per unit fraction;"
          f" radius 0.01 = {0.01 * a_per_frac[0]:.1f} A, 0.05 = {0.05 * a_per_frac[0]:.1f} A")
    n_multi = sum(len({q.parent for q in v}) > 1 for v in star_idx.values())
    print(f"{len(star_idx)} micrograph ids with star files under {args.star_root}"
          f" ({n_multi} appear in more than one directory)")

    # --- full per-pick distances at a (near-)zero match radius: nearest particle regardless of hit ---
    df = label_all(cs, mic_ids, a_per_frac, star_idx, match_radius_A=np.inf, ambiguity_radius_A=0.0, mic_dim=args.mic_dim)
    df = df.rename(columns={"label": "species1"})  # at match_radius_A=inf, label == species_nearest except for NoGT
    df.to_csv(args.out / "pick_distances.csv", index=False)
    print(f"wrote {args.out / 'pick_distances.csv'}")

    # --- label counts and ambiguity as a function of the radius -------------------------------
    rows = []
    for r in args.radii:
        rA = r * a_per_frac[0]
        hit = df["dist_A"].to_numpy() < rA
        lab = np.where(hit, df["species_nearest"], "Empty")
        amb = hit & (df["dist_other_A"].to_numpy() < rA)
        rows.append({"radius_frac": r, "radius_A": rA,
                     **{s: int((lab == s).sum()) for s in sorted(set(df["species_nearest"]) - {"NoGT"})},
                     "Empty": int((lab == "Empty").sum()), "NoGT": int((df["species_nearest"] == "NoGT").sum()),
                     "ambiguous": int(amb.sum())})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(args.out / "label_counts_by_radius.csv", index=False)

    # --- final labels: radius, minus ambiguous picks --------------------------------------------
    rA = args.radius * a_per_frac[0]
    ambiguity_radius_A = args.ambiguity_radius_A if args.ambiguity_radius_A is not None else rA
    lab_df = label_all(cs, mic_ids, a_per_frac, star_idx, match_radius_A=rA, ambiguity_radius_A=ambiguity_radius_A,
                        mic_dim=args.mic_dim)[["uid", "mic_id", "label"]]
    fn = args.out / f"labels_r{args.radius}_other{ambiguity_radius_A:.0f}A.csv"
    lab_df.to_csv(fn, index=False)
    print(f"\nradius {args.radius} ({rA:.1f} A), ambiguous if another species is closer than {ambiguity_radius_A:.1f} A:")
    print(lab_df["label"].value_counts().to_string())
    print(f"wrote {fn}")
    print("\nkept picks if the other-species distance must exceed (A), at radius", args.radius)
    hit = df["dist_A"].to_numpy() < rA
    sweep = {om: {s_: int(((df["species_nearest"] == s_) & hit & (df["dist_other_A"] >= om)).sum())
                  for s_ in sorted(set(df['species_nearest']) - {'NoGT'})} for om in (0, 15, 30, 45, 60, 90)}
    print(pd.DataFrame(sweep).T.rename_axis("other_min_A").to_string())
    print("\nnearest-particle distance (A), percentiles by nearest species:")
    print(df.groupby("species_nearest")["dist_A"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(1).to_string())

    # --- plots ----------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping plots")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    top = np.nanpercentile(df["dist_A"], 99.5)
    bins = np.linspace(0, top, 120)
    for s, g in df.groupby("species_nearest"):
        if s != "NoGT":
            ax.hist(g["dist_A"].dropna(), bins=bins, histtype="step", label=s)
    for r in args.radii:
        ax.axvline(r * a_per_frac[0], color="grey", ls=":", lw=0.8)
        ax.text(r * a_per_frac[0], ax.get_ylim()[1] * 0.97, f"{r}", rotation=90, va="top", ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("distance from pick to nearest simulated particle (A)")
    ax.set_ylabel("picks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "pick_nearest_distance.png", dpi=200)
    print(f"wrote {args.out / 'pick_nearest_distance.png'}")

    if args.refine_cs is not None:
        rf = np.load(args.refine_cs, allow_pickle=False)
        shift_A = np.linalg.norm(rf["alignments3D/shift"], axis=1) * rf["alignments3D/psize_A"]
        m = df.merge(pd.DataFrame({"uid": rf["uid"], "shift_A": shift_A}), on="uid")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for s, g in m.groupby("species_nearest"):
            if s != "NoGT":
                ax.scatter(g["dist_A"], g["shift_A"], s=3, alpha=0.3, label=s)
        lim = float(np.nanmax(m[["dist_A", "shift_A"]].to_numpy()))
        ax.plot([0, lim], [0, lim], "k:", lw=0.8)
        ax.set_xlabel("pick to nearest particle (A)")
        ax.set_ylabel("|refined shift| (A)")
        ax.legend(markerscale=4)
        fig.tight_layout()
        fig.savefig(args.out / "shift_vs_pick_offset.png", dpi=200)
        print(f"wrote {args.out / 'shift_vs_pick_offset.png'} ({len(m)} particles joined by uid)")


if __name__ == "__main__":
    main()
