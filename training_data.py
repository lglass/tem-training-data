"""Collect and label training data from CryoSPARC particle-picking jobs.

Walks every job of a given type (default: downsample_particles) in a
project/workspace, matches each extracted particle back to the TEM-simulator
ground truth (the .star files txt2star.py produces) by pick location within
`match_radius_A` (Angstrom; the deprecated `match_radius` fraction-of-width
field still works, converted at run time). The matched species, or "Empty"
(nothing within radius), "Ambiguous" (another species also within
`ambiguity_radius_A`), or "NoGT" (no ground truth at all for that
micrograph) -- see labeling.py -- is mapped via `label_map` to an integer
class label, and a class-balanced CSV is written.

    python training_data.py --config configs/training_data.yaml
    python training_data.py --config configs/training_data.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import starfile
from cryosparc.tools import CryoSPARC

import labeling
from config import TrainingDataConfig, load_training_data_config


def get_mic_dir(project, job_id, param_name="blob_paths"):
    job = project.find_job(job_id)

    # 1. Safely traverse the job document
    # Most Import jobs store the path in params_spec -> [param_name] -> value
    params_spec = job.doc.get("params_spec", {})
    param_entry = params_spec.get(param_name, {})
    blob_path_str = param_entry.get("value")

    if not blob_path_str:
        # Fallback/Error: Some jobs might use different keys or the job hasn't been built
        raise ValueError(
            f"Could not find a path value for '{param_name}' in job {job_id}. "
            "Double-check the job type or parameter name."
        )

    # 2. Convert to Path object and expand any ~ symbols
    p = Path(blob_path_str).expanduser()

    # 3. Robust Path Logic:
    # If the last part of the path (p.name) contains a wildcard (*)
    # OR has a file extension (p.suffix), it's a file/pattern.
    # In those cases, we want the .parent (the directory).
    if "*" in p.name or p.suffix:
        mic_dir = p.parent
    else:
        # It's likely already a directory (e.g., "/data/micrographs")
        mic_dir = p

    return mic_dir


def read_cryosparc_particles(project, job_id):
    job = project.find_job(job_id)
    # dataset.to_pandas() is usually more efficient in cryosparc-tools
    return pd.DataFrame(job.load_output("particles").rows())


def read_coords(mrc_dir: Path, mic_id: str, mic_dim: int):
    """Read the ground-truth star file(s) for one micrograph. Pure I/O: no
    filtering or coordinate conversion -- see labeling.GroundTruth.from_coords
    for that. Raises FileNotFoundError if no matching star file exists."""
    # Flexible glob to handle potential UID prefixes
    star_paths = list(mrc_dir.glob(f"*coord_{mic_id}_*.star"))

    if not star_paths:
        raise FileNotFoundError(
            f"No coordinate files found for mic_id {mic_id} in {mrc_dir}"
        )

    return pd.concat([starfile.read(p) for p in star_paths], ignore_index=True)


_warned_deprecated_match_radius = False


def _resolve_match_radius_A(
    match_radius: Optional[float], match_radius_A: Optional[float], a_per_frac: float
) -> float:
    global _warned_deprecated_match_radius
    if match_radius_A is not None:
        return match_radius_A
    radius_A = match_radius * a_per_frac
    if not _warned_deprecated_match_radius:
        warnings.warn(
            "'match_radius' (fraction of micrograph width) is deprecated; converting to "
            f"Angstrom via this job's pixel size ({match_radius} * {a_per_frac:.1f} = "
            f"{radius_A:.1f} A for this micrograph). Set 'match_radius_A' directly to "
            "silence this warning.",
            DeprecationWarning,
        )
        _warned_deprecated_match_radius = True
    return radius_A


def label_particles(
    particles: pd.DataFrame,
    mrc_dir: Union[Path, str],
    mic_dim: int,
    match_radius: Optional[float],
    match_radius_A: Optional[float],
    ambiguity_radius_A: float,
) -> pd.DataFrame:
    """Label every pick in `particles` against the TEM-simulator ground truth,
    via the shared rule in labeling.py. Adds columns: particle (the label:
    species name, "Empty", "Ambiguous" or "NoGT"), species_nearest, dist_A,
    dist_other_A, rlnAngleRot/Tilt/Psi (of the nearest particle), dx_A, dy_A.
    """
    particles = particles.copy()
    mrc_dir = Path(mrc_dir).expanduser()

    a_per_frac = labeling.angstrom_per_frac(
        np.stack(particles["location/micrograph_shape"].to_numpy()),
        particles["location/micrograph_psize_A"].to_numpy(),
    )

    n = len(particles)
    cols = {
        "particle": np.full(n, labeling.NO_GT, dtype=object),
        "species_nearest": np.full(n, labeling.NO_GT, dtype=object),
        "dist_A": np.full(n, np.nan), "dist_other_A": np.full(n, np.nan),
        "rlnAngleRot": np.full(n, np.nan), "rlnAngleTilt": np.full(n, np.nan), "rlnAnglePsi": np.full(n, np.nan),
        "dx_A": np.full(n, np.nan), "dy_A": np.full(n, np.nan),
    }

    for mic_path, grp in particles.groupby("location/micrograph_path"):
        mic_id = Path(str(mic_path)).stem.rsplit("_", 1)[-1]  # ID after last underscore
        indices = grp.index
        pos = particles.index.get_indexer(indices)
        radius_A = _resolve_match_radius_A(match_radius, match_radius_A, a_per_frac[pos[0]])

        try:
            star_df = read_coords(mrc_dir, mic_id, mic_dim)
            gt = labeling.GroundTruth.from_coords(
                star_df["rlnCoordinateX"].to_numpy(), star_df["rlnCoordinateY"].to_numpy(),
                star_df["rlnStructure"].to_numpy(),
                star_df["rlnAngleRot"].to_numpy(), star_df["rlnAngleTilt"].to_numpy(), star_df["rlnAnglePsi"].to_numpy(),
                mic_dim,
            )
        except FileNotFoundError:
            gt = None

        result = labeling.label_picks(
            grp["location/center_x_frac"].to_numpy(), grp["location/center_y_frac"].to_numpy(),
            gt, a_per_frac=a_per_frac[pos[0]], match_radius_A=radius_A, ambiguity_radius_A=ambiguity_radius_A,
        )
        cols["particle"][pos] = result.label
        cols["species_nearest"][pos] = result.species_nearest
        cols["dist_A"][pos] = result.dist_A
        cols["dist_other_A"][pos] = result.dist_other_A
        cols["rlnAngleRot"][pos] = result.rot
        cols["rlnAngleTilt"][pos] = result.tilt
        cols["rlnAnglePsi"][pos] = result.psi
        cols["dx_A"][pos] = result.dx_A
        cols["dy_A"][pos] = result.dy_A

    for name, values in cols.items():
        particles[name] = values
    return particles


def get_jobs_with_particles(
    cryosparc: CryoSPARC,
    project_uid: str,
    workspace_uid: str,
    job_type: str = "extract_micrographs_cpu_parallel",
):
    job_uids = []
    for job in cryosparc.cli.get_jobs_by_type(
        project_uid=project_uid, workspace_uid=workspace_uid, types=[job_type]
    ):
        job_uids.append(job["uid"])
    return job_uids


def get_import_job(cryosparc: CryoSPARC, project, job_uid: str):
    for anc_job_id in cryosparc.cli.job_find_ancestors(
        project_uid=project.uid, job_uid=job_uid
    ):
        anc_job = project.find_job(anc_job_id)
        if anc_job.type == "import_micrographs":
            return anc_job.uid
    return None


def structure2label(training: pd.DataFrame, label_map: dict[str, int]) -> pd.DataFrame:
    training = training.copy()
    mapped = training["particle"].map(label_map)
    unmapped = sorted(training.loc[mapped.isna(), "particle"].unique())
    if unmapped:
        raise ValueError(f"label_map is missing entries for: {unmapped}")
    training["label"] = mapped.astype(int)
    return training


def collect_training_data(cs: CryoSPARC, cfg: TrainingDataConfig) -> pd.DataFrame:
    project = cs.find_project(cfg.project)
    training_data = []

    for job_id in get_jobs_with_particles(
        cs, cfg.project, cfg.workspace, job_type=cfg.particle_job_type
    ):
        import_job_id = get_import_job(cs, project, job_id)
        print(f"Found job with particles: {job_id}, import job: {import_job_id}")
        mic_dir = get_mic_dir(project, import_job_id)
        try:
            particles_df = read_cryosparc_particles(project, job_id)
            particles_df["blob_path"] = Path(project.dir()) / particles_df["blob/path"]
        except Exception as e:
            print(f"Error reading particles for job {job_id}: {e}")
            continue
        particles_df_labeled = label_particles(
            particles_df, mrc_dir=mic_dir, mic_dim=cfg.mic_dim,
            match_radius=cfg.match_radius, match_radius_A=cfg.match_radius_A,
            ambiguity_radius_A=cfg.ambiguity_radius_A,
        )
        particles_df_labeled["source_job_id"] = job_id
        training_data.append(particles_df_labeled)

    print(f"Simulated data collected: {len(training_data)} particle sets")

    # Combine all labeled data into a single DataFrame
    return pd.concat(training_data, ignore_index=True)


def connect(cfg: TrainingDataConfig) -> CryoSPARC:
    creds = cfg.connection.load()
    cs = CryoSPARC(
        host=creds["CRYOSPARC_HOST"],
        base_port=int(creds["CRYOSPARC_PORT"]),
        license=creds["CRYOSPARC_LICENSE_ID"],
        email=creds["CRYOSPARC_EMAIL"],
        password=creds["CRYOSPARC_PASSWORD"],
    )
    if not cs.test_connection():
        raise RuntimeError("Could not connect to CryoSPARC")
    return cs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training data from CryoSPARC project"
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Override output_csv from the config."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample_size (per-class cap) from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and label particles, print per-label counts, and write a sidecar "
        "<output_csv>.json with those counts and the run's parameters -- but do not "
        "sample or write output_csv.",
    )
    args = parser.parse_args()

    cfg = load_training_data_config(args.config)
    overrides = {
        key: value
        for key, value in (("output_csv", args.output), ("sample_size", args.sample_size))
        if value is not None
    }
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    cs = connect(cfg)

    training_data = collect_training_data(cs, cfg)
    training_data = structure2label(training_data, cfg.label_map)

    if args.dry_run:
        counts_by_name = training_data["particle"].value_counts().to_dict()
        counts_by_int = {
            str(k): int(v) for k, v in training_data["label"].value_counts().sort_index().items()
        }
        sidecar = {
            "params": {
                "project": cfg.project, "workspace": cfg.workspace,
                "particle_job_type": cfg.particle_job_type,
                "match_radius": cfg.match_radius, "match_radius_A": cfg.match_radius_A,
                "ambiguity_radius_A": cfg.ambiguity_radius_A, "mic_dim": cfg.mic_dim,
                "label_map": cfg.label_map,
            },
            "counts_by_label_name": counts_by_name,
            "counts_by_label_int": counts_by_int,
            "job_ids": sorted(training_data["source_job_id"].unique().tolist()),
            "total_rows": int(len(training_data)),
        }
        sidecar_path = Path(str(cfg.output_csv) + ".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))
        print(f"[dry-run] counts: {counts_by_name}")
        print(f"[dry-run] wrote {sidecar_path}; nothing written to {cfg.output_csv}")
        return

    training_data = (
        training_data.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cfg.sample_size)))
        .sort_index()
    )
    training_data.to_csv(cfg.output_csv, index=False)
    print(f"Wrote {len(training_data)} rows to {cfg.output_csv}")


if __name__ == "__main__":
    main()
