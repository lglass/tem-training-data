import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from cryosparc.tools import CryoSPARC
from cryosparc.util import first
from dotenv import load_dotenv

experiments_dir = Path("/mnt/data/lysate/")

load_dotenv(".env")
CRYOSPARC_LICENSE_ID = os.getenv("CRYOSPARC_LICENSE_ID")
CRYOSPARC_EMAIL = os.getenv("CRYOSPARC_EMAIL")
CRYOSPARC_PASSWORD = os.getenv("CRYOSPARC_PASSWORD")
CRYOSPARC_HOST = os.getenv("CRYOSPARC_HOST")
CRYOSPARC_PORT = int(os.getenv("CRYOSPARC_PORT"))

PROJECT = "P8"
WORKSPACE = "W2"

cs = CryoSPARC(
    host=CRYOSPARC_HOST,
    base_port=CRYOSPARC_PORT,
    license=CRYOSPARC_LICENSE_ID,
    email=CRYOSPARC_EMAIL,
    password=CRYOSPARC_PASSWORD,
)

assert cs.test_connection()

project = cs.find_project(PROJECT)
workspace = project.find_workspace(WORKSPACE)
lane = "default"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CryoSPARC micrograph import and refinement pipeline."
    )
    parser.add_argument(
        "--dose",
        type=float,
        default=30.0,
        help="Fallback total electron dose in e/A^2 used for micrograph import.",
    )
    parser.add_argument(
        "--mrc-dirs-file",
        default="mrc_dirs.txt",
        help="Path to file containing one micrograph directory per line.",
    )
    parser.add_argument(
        "--mrc-dose-map",
        default="mrc_dirs_dose_map.json",
        help="Optional JSON file mapping directory paths to dose values.",
    )
    return parser.parse_args()


def resolve_dose_from_dir(data_dir: Path) -> float | None:
    """Extract dose from directory name like dose_45.0."""
    match = re.search(r"dose_(-?\d+(?:\.\d+)?)", data_dir.as_posix())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


args = parse_args()

log_file_path = datetime.now().strftime("cs_run_%Y-%m-%d %H:%M.txt")

with open(args.mrc_dirs_file, "r") as f:
    data_dirs = f.read().splitlines()

dose_map: dict[str, float] = {}
map_path = Path(args.mrc_dose_map)
if map_path.exists():
    with open(map_path, "r") as f:
        raw_map = json.load(f)
    for key, value in raw_map.items():
        try:
            dose_map[str(Path(key))] = float(value)
        except (TypeError, ValueError):
            continue

for data_dir in data_dirs:
    data_dir = Path(data_dir)

    blob_pattern = (data_dir / "*.mrc").as_posix()
    has_micrographs = any(data_dir.glob("*.mrc"))
    if not has_micrographs:
        print(f"Skipping empty directory: {data_dir}")
        continue

    resolved_dose = dose_map.get(str(data_dir))
    if resolved_dose is None:
        resolved_dose = resolve_dose_from_dir(data_dir)
    if resolved_dose is None:
        resolved_dose = args.dose
        print(
            "WARNING: Could not resolve dose from map/path "
            f"for {data_dir}. Falling back to --dose={args.dose}."
        )

    import_micrographs_job = workspace.create_job(
        "import_micrographs",
        params={
            "blob_paths": blob_pattern,
            "psize_A": 1.5678,
            "accel_kv": 200,
            "cs_mm": 2.7,
            "total_dose_e_per_A2": resolved_dose,
        },
    )

    import_micrographs_job.queue(lane)

    ###
    ctf_estimation_job = workspace.create_job(
        "ctf_estimation",
        connections={"exposures": (import_micrographs_job.uid, "imported_micrographs")},
        params={
            "astigmatism_known": True,
            "spectrum_dim": 1024,
            "min_res_A": 30,
            "max_res_A": 5,
            "min_df_A": 10000,
            "max_df_A": 40000,
            "step_df_A": 100,
            "known_astigmatism_A": 0,
            "slow": False,
        },
    )

    ctf_estimation_job.queue(lane)

    ###
    curate_exposures_job = workspace.create_job(
        "curate_exposures_v2",
        connections={"exposures": (ctf_estimation_job.uid, "exposures_success")},
    )

    curate_exposures_job.queue()
    curate_exposures_job.wait_for_status("waiting")

    data = curate_exposures_job.interact("get_fields_and_thresholds")

    idx_field = first(
        field for field in data["fields"] if field["name"] == "ctf_fit_to_A"
    )
    assert idx_field
    idx_field["thresholds"] = [0, 16]
    idx_field["active"] = True

    curate_exposures_job.interact("set_thresholds", data)
    curate_exposures_job.interact("shutdown_interactive")

    ###
    particle_diameters = [130, 270]  # in Angstroms

    topaz_job_ids = []
    # For one particle diameter, we can directly extract particles. For multiple diameters, we first merge the particles from different diameters and then extract them together, dropping duplicate particles.
    if len(particle_diameters) == 1:
        topaz_job = workspace.create_job(
            "topaz_extract",
            connections={"micrographs": (curate_exposures_job.uid, "exposures_accepted")},
            params={
                "exec_path": "/home/cryosparcuser/topaz.sh",
                "pretrained": "ResNet16 (32 units)",
                "par_diam": particle_diameters[0],
                "threshold": 1,
            },
        )
        topaz_job.queue(lane)
        topaz_job.wait_for_done()

        extract_job = workspace.create_job(
            "extract_micrographs_cpu_parallel",
            connections={
                "micrographs": (topaz_job.uid, "micrographs"),
                "particles": (topaz_job.uid, "particles"),
            },
            params={"box_size_pix": 384},
        )
        extract_job.queue(lane)
    else:
        for par_diam in particle_diameters:
            topaz_job = workspace.create_job(
                "topaz_extract",
                connections={"micrographs": (curate_exposures_job.uid, "exposures_accepted")},
                params={
                    "exec_path": "/home/cryosparcuser/topaz.sh",
                    "pretrained": "ResNet16 (32 units)",
                    "par_diam": par_diam,
                    "threshold": 1,
                },
            )
            topaz_job.queue(lane)
            topaz_job.wait_for_done()


    ###
    # extract_job = workspace.create_job(
    #     "extract_micrographs_cpu_parallel",
    #     connections={
    #         "micrographs": (topaz_job_small.uid, "micrographs"),
    #         "particles": (topaz_job_small.uid, "particles"),
    #     },
    #     params={"box_size_pix": 384},
    # )
    # extract_job.queue(lane)

    # downsample_job = workspace.create_job(
    #     "downsample_particles",
    #     connections={"particles": (extract_job.uid, "particles")},
    #     params={"bin_size_pix": 160},
    # )
    # downsample_job.queue(lane)

    ###
    # het_refine_job = workspace.create_job(
    #     "hetero_refine",
    #     connections={
    #         "particles": (downsample_job.uid, "particles"),
    #         "volume": [(VOLUME_JOB, f"imported_volume_{v}") for v in range(1, 4 + 1)],
    #     },
    # )
    # het_refine_job.queue(lane)
    # het_refine_job.wait_for_done()

    # homo_refine_jobs = []
    # for vol_num in range(0, 3 + 1):
    #     homo_refine_job = workspace.create_job(
    #         type="homo_refine_new",
    #         connections={
    #             "particles": (het_refine_job.uid, f"particles_class_{vol_num}"),
    #             "volume": (het_refine_job.uid, f"volume_class_{vol_num}"),
    #         },
    #     )
    #     homo_refine_job.connect_result(
    #         target_input="particles",
    #         connection_idx=0,
    #         slot="blob",
    #         source_job_uid=extract_job.uid,
    #         source_output="particles",
    #         source_result="blob",
    #     )
    #     homo_refine_job.queue("default")
    #     homo_refine_jobs.append(homo_refine_job.uid)

