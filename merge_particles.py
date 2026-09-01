import argparse
from pathlib import Path

from utils_cs import initialize_cryosparc, load_env_variables

PROJECT = "P8"
LANE = "default"
PROJECT_DIR = Path("/mnt/data/CS-empiar-10892/")
PARTICLE_OUTPUT = "particles"
DEDUP_KEYS = [
    "location/micrograph_uid",
    "location/center_x_frac",
    "location/center_y_frac",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge particle datasets from CryoSPARC jobs."
    )
    parser.add_argument(
        "job_ids",
        nargs="+",
        help="CryoSPARC job ids to merge, in priority order.",
    )
    parser.add_argument(
        "--output-name",
        default="particles",
        help="Name of the external result to create.",
    )
    return parser.parse_args()


def load_particle_dataset(project, job_id, output_name=PARTICLE_OUTPUT):
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


def main():
    args = parse_args()

    cs = initialize_cryosparc(load_env_variables("/home/lara/.env"))
    project = cs.find_project(PROJECT)
    workspace = project.find_workspace("W2")

    particle_sets = [
        load_particle_dataset(project, job_id, output_name=PARTICLE_OUTPUT)
        for job_id in args.job_ids
    ]
    merged_particles = merge_particle_datasets(particle_sets, DEDUP_KEYS)

    job_id = project.save_external_result(
        workspace_uid=workspace.uid,
        dataset=merged_particles,
        type="particle",
        name=args.output_name,
    )

    print(f"Saved merged particle dataset as external result {job_id}")


if __name__ == "__main__":
    main()