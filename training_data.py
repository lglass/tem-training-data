import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import starfile
from cryosparc.tools import CryoSPARC
from scipy.spatial import cKDTree

from utils_cs import initialize_cryosparc, load_env_variables

NONE_LABEL = 0  # Label for "Empty" particles (keep consistent with mapping)


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
    # Flexible glob to handle potential UID prefixes
    star_paths = list(mrc_dir.glob(f"*coord_{mic_id}_*.star"))

    if not star_paths:
        raise FileNotFoundError(
            f"No coordinate files found for mic_id {mic_id} in {mrc_dir}"
        )

    coords_gt = pd.concat([starfile.read(p) for p in star_paths], ignore_index=True)

    coords_gt = coords_gt[coords_gt["rlnCoordinateX"].between(0, mic_dim)]
    coords_gt = coords_gt[coords_gt["rlnCoordinateY"].between(0, mic_dim)]

    # Calculate fractional coordinates
    coords_gt["location/center_x_frac"] = (
        coords_gt["rlnCoordinateX"].astype(float) / mic_dim
    )
    coords_gt["location/center_y_frac"] = (
        coords_gt["rlnCoordinateY"].astype(float) / mic_dim
    )

    return coords_gt.rename(columns={"rlnStructure": "particle"})


def assign_structure_view(
    xys: np.ndarray, mrc_dir: Path, mic_id: str, radius: float, mic_dim: int
):
    try:
        coords_gt = read_coords(mrc_dir, mic_id, mic_dim)
    except FileNotFoundError:
        # Return "Empty" if no ground truth exists for this micrograph
        return np.full(len(xys), "Empty", dtype=object)

    gt_xy = coords_gt[["location/center_x_frac", "location/center_y_frac"]].values

    # Use KD-Tree for fast spatial matching
    tree = cKDTree(gt_xy)
    distances, indices = tree.query(xys, distance_upper_bound=radius)

    # results from query: if no neighbor in radius, distance is inf, index is len(gt_xy)
    has_hit = distances < radius

    structure = np.full(len(xys), "Empty", dtype=object)

    # Assign labels where hits were found
    valid_indices = indices[has_hit]
    structure[has_hit] = coords_gt.iloc[valid_indices]["particle"].values

    return structure


def assign_structures_views(
    data, mrc_dir: Optional[Union[Path, str]] = None, radius: float = 0.01, mic_dim: int = 4096
):
    data = data.copy()

    if mrc_dir is None:
        raise ValueError("mrc_dir must be provided to assign_structures_views")

    mrc_dir = Path(mrc_dir).expanduser()

    # Pre-allocate columns to avoid fragmented dataframe warnings
    data["particle"] = "Empty"
    data["label"] = NONE_LABEL

    # Group by micrograph to process one KD-Tree at a time
    for mic_path, grp in data.groupby("location/micrograph_path"):
        mic_id = Path(str(mic_path)).stem.rsplit("_", 1)[-1]  # ID after last underscore

        indices = grp.index
        xys = grp[["location/center_x_frac", "location/center_y_frac"]].values

        structs = assign_structure_view(xys, mrc_dir, mic_id, radius, mic_dim)

        data.loc[indices, "particle"] = structs

    return data


def assign_all_labels(
    data,
    mrc_dir: Optional[Union[Path, str]] = None,
    radius: Optional[float] = 0.01,
    mic_dim: Optional[int] = 4096,
):
    data = assign_structures_views(
        data, mrc_dir=mrc_dir, radius=radius, mic_dim=mic_dim
    )
    return data


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


def structure2label(training: pd.DataFrame):
    id2label = {
        "Empty": NONE_LABEL,
        "7Q5Q": 1,
        "7Q5R": 2,
        "7Q5S": 3,
        "3JCT": 4,
        "6ELZ": 5,
        "6EM5": 6,
    }

    training["label"] = training["particle"].map(id2label).fillna(NONE_LABEL).astype(int)
    return training


def main(args):

    cs = initialize_cryosparc(load_env_variables(".env"))

    project = cs.find_project(args.project_id)
    training_data = []

    # Simulated

    for job_id in get_jobs_with_particles(
        cs, args.project_id, args.workspace_id, job_type="downsample_particles"
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
        # Assign structure and view labels based on coordinates
        # and map those to the final class labels
        particles_df_labeled = assign_all_labels(
            particles_df, mrc_dir=mic_dir, radius=0.05, mic_dim=4096
        )
        training_data.append(particles_df_labeled)

    print(f"Simulated data collected: {len(training_data)} particle sets")

    # Combine all labeled data into a single DataFrame
    training_data = pd.concat(training_data, ignore_index=True)

    return training_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect training data from CryoSPARC project"
    )
    parser.add_argument(
        "--project-id",
        default="P4",
        help="CryoSPARC project ID to collect data from.",
    )
    parser.add_argument(
        "--workspace-id",
        default="W4",
        help="CryoSPARC workspace ID to collect data from.",
    )
    parser.add_argument(
        "--output", default="training.csv", help="Output CSV file for training data."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=70000,
        help="Number of samples per class to keep.",
    )

    args = parser.parse_args()

    training_data = main(args)

    training_data = structure2label(training_data)

    training_data = (
        training_data.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.sample_size)))
        .sort_index()
    )
    training_data.to_csv(args.output, index=False)
