"""Collect experimental (non-simulated) particle picks from CryoSPARC jobs
and label them by species for supervised training.

Each job listed in ``particles`` is assumed to contain particles of a single
known species (e.g. a topaz_extract run targeted at one protein). Unlike
training_data.py's per-particle ground-truth matching against simulated
coordinates, every particle pulled from a job simply gets that job's
configured label.

    python experiment_training.py --config configs/experiment_data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from cryosparc.tools import CryoSPARC

from config import ExperimentalDataConfig, load_exp_data_config


def get_particle_df(project, job_uid: str) -> pd.DataFrame:
    job = project.find_job(job_uid)
    try:
        ds = job.load_output("particles").filter_fields(
            ["uid", "blob/path", "blob/idx"]
        )
    except Exception as e:
        print(f"Error loading particles for job {job_uid}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    return pd.DataFrame(ds.rows())


def process_jobs(project, cfg: ExperimentalDataConfig) -> pd.DataFrame:
    exp_ground_truth = []
    for particle in cfg.particles:
        df = get_particle_df(project, particle.job_uid)
        if df.empty:
            continue
        df["job_uid"] = particle.job_uid
        df["label"] = particle.label
        df["name"] = particle.name
        exp_ground_truth.append(df)

    if not exp_ground_truth:
        raise ValueError("No particles were collected from any configured job.")

    return pd.concat(exp_ground_truth, ignore_index=True)


def connect(cfg: ExperimentalDataConfig) -> CryoSPARC:
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
        description="Collect experimental particle data for supervised training."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Override output_csv from the config."
    )
    args = parser.parse_args()

    cfg = load_exp_data_config(args.config)
    if args.output is not None:
        cfg = cfg.model_copy(update={"output_csv": args.output})

    cs = connect(cfg)
    project = cs.find_project(cfg.project)

    training_data = process_jobs(project, cfg)
    training_data.to_csv(cfg.output_csv, index=False)
    print(f"Wrote {len(training_data)} rows to {cfg.output_csv}")


if __name__ == "__main__":
    main()
