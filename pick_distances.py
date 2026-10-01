"""Distance from every CryoSPARC pick to the nearest simulated (ground-truth) particle.

Reads
  * a CryoSPARC particle .cs file (numpy structured array; needs location/* fields), and
  * the TEM-simulator ground-truth coord_<micid>_<species>.star files (searched recursively),
and reports, per pick, the distance (in Angstrom) to the nearest ground-truth particle, its
species and Euler angles, and the distance/species of the second nearest. It then tabulates how
the labelling radius (a fraction of the micrograph width, as in training_data.py) changes the
species counts, and how many picks are ambiguous (nearest of the other species is also in range).

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
from scipy.spatial import cKDTree


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", required=True, type=Path)
    ap.add_argument("--star-root", required=True, type=Path)
    ap.add_argument("--mic-dim", type=int, default=4096, help="simulated micrograph size in px (as in training_data.py)")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05],
                    help="labelling radii as fractions of the micrograph width")
    ap.add_argument("--out", type=Path, default=Path("pick_distances_out"))
    ap.add_argument("--refine-cs", type=Path, default=None)
    ap.add_argument("--max-picks", type=int, default=None, help="subsample for a quick test")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cs = np.load(args.cs, allow_pickle=False)
    if args.max_picks and len(cs) > args.max_picks:
        cs = cs[np.random.default_rng(0).choice(len(cs), args.max_picks, replace=False)]
    print(f"{len(cs)} picks in {args.cs}")

    shape = cs["location/micrograph_shape"]
    psize = cs["location/micrograph_psize_A"].astype(float)
    if not np.all(shape[:, 0] == shape[:, 1]):
        raise SystemExit("non-square micrographs: fractional distance is not isotropic; extend the script")
    a_per_frac = shape[:, 1].astype(float) * psize  # Angstrom per unit of fractional coordinate
    print(f"micrograph {shape[0].tolist()} px at {psize[0]:.4f} A/px -> {a_per_frac[0]:.1f} A per unit fraction;"
          f" radius 0.01 = {0.01 * a_per_frac[0]:.1f} A, 0.05 = {0.05 * a_per_frac[0]:.1f} A")

    pick_frac = np.stack([cs["location/center_x_frac"], cs["location/center_y_frac"]], axis=1).astype(float)
    mic_ids = np.array([Path(decode(p)).stem.rsplit("_", 1)[-1] for p in cs["location/micrograph_path"]])

    star_idx = index_star_files(args.star_root)
    n_multi = sum(len({q.parent for q in v}) > 1 for v in star_idx.values())
    print(f"{len(star_idx)} micrograph ids with star files under {args.star_root}"
          f" ({n_multi} appear in more than one directory)")

    n = len(cs)
    out = {
        "dist_A": np.full(n, np.nan), "species1": np.full(n, "NoGT", dtype=object),
        "dist2_A": np.full(n, np.nan), "species2": np.full(n, "", dtype=object),
        "rot": np.full(n, np.nan), "tilt": np.full(n, np.nan), "psi": np.full(n, np.nan),
        "dx_A": np.full(n, np.nan), "dy_A": np.full(n, np.nan),
    }
    for mic in np.unique(mic_ids):
        sel = np.where(mic_ids == mic)[0]
        paths = star_idx.get(mic)
        if not paths:
            continue
        gt = pd.concat([read_star(p) for p in paths], ignore_index=True)
        gt = gt[gt["rlnCoordinateX"].between(0, args.mic_dim) & gt["rlnCoordinateY"].between(0, args.mic_dim)]
        gt = gt.reset_index(drop=True)
        if len(gt) == 0:
            continue
        gt_frac = gt[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy() / args.mic_dim
        tree = cKDTree(gt_frac)
        k = min(2, len(gt))
        d, i = tree.query(pick_frac[sel], k=k)
        d = d.reshape(len(sel), k)
        i = i.reshape(len(sel), k)
        scale = a_per_frac[sel]
        out["dist_A"][sel] = d[:, 0] * scale
        out["species1"][sel] = gt["rlnStructure"].to_numpy()[i[:, 0]]
        out["rot"][sel] = gt["rlnAngleRot"].to_numpy()[i[:, 0]]
        out["tilt"][sel] = gt["rlnAngleTilt"].to_numpy()[i[:, 0]]
        out["psi"][sel] = gt["rlnAnglePsi"].to_numpy()[i[:, 0]]
        out["dx_A"][sel] = (gt_frac[i[:, 0], 0] - pick_frac[sel, 0]) * scale
        out["dy_A"][sel] = (gt_frac[i[:, 0], 1] - pick_frac[sel, 1]) * scale
        if k == 2:
            out["dist2_A"][sel] = d[:, 1] * scale
            out["species2"][sel] = gt["rlnStructure"].to_numpy()[i[:, 1]]

    df = pd.DataFrame({"uid": cs["uid"], "mic_id": mic_ids, **out})
    df.to_csv(args.out / "pick_distances.csv", index=False)
    print(f"wrote {args.out / 'pick_distances.csv'}")

    # --- label counts and ambiguity as a function of the radius -------------------------------
    rows = []
    for r in args.radii:
        rA = r * a_per_frac
        hit = df["dist_A"].to_numpy() < rA
        lab = np.where(hit, df["species1"], "Empty")
        # ambiguous: the second-nearest particle is of the other species and also within r
        amb = hit & (df["dist2_A"].to_numpy() < rA) & (df["species1"] != df["species2"])
        rows.append({"radius_frac": r, "radius_A": rA[0],
                     **{s: int((lab == s).sum()) for s in sorted(set(df["species1"]) - {"NoGT"})},
                     "Empty": int((lab == "Empty").sum()), "NoGT": int((df["species1"] == "NoGT").sum()),
                     "ambiguous": int(amb.sum())})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(args.out / "label_counts_by_radius.csv", index=False)
    print("\nnearest-particle distance (A), percentiles by nearest species:")
    print(df.groupby("species1")["dist_A"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(1).to_string())

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
    for s, g in df.groupby("species1"):
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
        for s, g in m.groupby("species1"):
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
