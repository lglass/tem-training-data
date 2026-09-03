"""Merge particle picks from multiple CryoSPARC jobs, deduplicating by pick
location.

Typical use: several topaz_extract/extract_micrographs jobs were run at
different particle diameters (see cryosparc_run.run_import's multi-diameter
branch, which does not merge across diameters), and a single, deduplicated
particle set is needed for downstream processing.

    python merge_particles.py --config configs/merge_particles.yaml
    python merge_particles.py --config configs/merge_particles.yaml J10 J11 J12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryosparc.tools import CryoSPARC

from config import MergeParticlesConfig, load_merge_particles_config


def load_particle_dataset(project, job_id, output_name="particles"):
    return project.find_job(job_id).load_output(output_name)


def merge_particle_datasets(datasets, dedup_keys):
    selected_indices_by_dataset = [[] for _ in datasets]
    seen_keys = set()

    for dataset_index, dataset in enumerate(datasets):
        for row_index, row in enumerate(dataset.rows()):
            key = tuple(row[key_name] for key_name in dedup_keys)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_indices_by_dataset[dataset_index].append(row_index)

    merged_particles = None
    for dataset, row_indices in zip(datasets, selected_indices_by_dataset):
        if not row_indices:
            continue

        selected_particles = dataset.take(row_indices)
        if merged_particles is None:
            merged_particles = selected_particles
        else:
            merged_particles = merged_particles.append(selected_particles)

    if merged_particles is None:
        raise ValueError("No particle rows were selected from the provided datasets.")

    return merged_particles


def run_merge(cfg: MergeParticlesConfig, job_ids: list[str]) -> str:
    if not job_ids:
        raise ValueError("No job ids provided (set `job_ids` in the config or pass them on the CLI).")

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

    project = cs.find_project(cfg.project)
    workspace = project.find_workspace(cfg.workspace)

    particle_sets = [
        load_particle_dataset(project, job_id, output_name="particles") for job_id in job_ids
    ]
    merged_particles = merge_particle_datasets(particle_sets, cfg.dedup_keys)

    return project.save_external_result(
        workspace_uid=workspace.uid,
        dataset=merged_particles,
        type="particle",
        name=cfg.output_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge particle datasets from CryoSPARC jobs, deduplicating by pick location."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "job_ids",
        nargs="*",
        help="CryoSPARC job ids to merge, in priority order. Overrides `job_ids` in the config.",
    )
    parser.add_argument(
        "--output-name", default=None, help="Override output_name from the config."
    )
    args = parser.parse_args()

    cfg = load_merge_particles_config(args.config)
    if args.output_name is not None:
        cfg = cfg.model_copy(update={"output_name": args.output_name})

    job_ids = args.job_ids or cfg.job_ids

    result_id = run_merge(cfg, job_ids)
    print(f"Saved merged particle dataset as external result {result_id}")


if __name__ == "__main__":
    main()
