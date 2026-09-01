"""Import simulated micrographs into CryoSPARC and run the standard
import -> CTF -> curate -> topaz pipeline.

Callable two ways:
  - standalone: `python cryosparc_run.py --config configs/cryosparc.yaml`
  - chained from run_sim_parallel.py, which calls run_import() directly with
    the mrc directories and dose map it just produced.

Connection credentials are never stored in the YAML config: they are loaded
from an env file (see config.CryoSPARCConnectionConfig) at the path the
config points to.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import yaml
from cryosparc.tools import CryoSPARC
from cryosparc.util import first

from config import CryoSPARCImportConfig, MicroscopeConfig, resolve_microscope_spec


def resolve_dose_from_dir(data_dir: Path) -> float | None:
    """Extract dose from a directory name like dose_45.0 (fallback only)."""
    match = re.search(r"dose_(-?\d+(?:\.\d+)?)", data_dir.as_posix())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def run_import(
    cfg: CryoSPARCImportConfig,
    microscope: MicroscopeConfig | None,
    mrc_dirs: Sequence[Path],
    dose_map: dict[str, float],
) -> None:
    if microscope is not None:
        cfg = cfg.with_microscope_derived_fields(microscope)

    missing = [f for f in ("pixel_size_A", "accel_kv", "cs_mm") if getattr(cfg, f) is None]
    if missing:
        raise ValueError(
            f"cryosparc config is missing {missing}; either provide a `microscope:` "
            "to derive them from, or set them explicitly."
        )

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

    for data_dir in mrc_dirs:
        data_dir = Path(data_dir)

        blob_pattern = (data_dir / "*.mrc").as_posix()
        if not any(data_dir.glob("*.mrc")):
            print(f"Skipping empty directory: {data_dir}")
            continue

        resolved_dose = dose_map.get(str(data_dir))
        if resolved_dose is None:
            resolved_dose = resolve_dose_from_dir(data_dir)
        if resolved_dose is None:
            resolved_dose = cfg.dose_fallback
            print(
                "WARNING: Could not resolve dose from map/path "
                f"for {data_dir}. Falling back to dose_fallback={cfg.dose_fallback}."
            )

        import_micrographs_job = workspace.create_job(
            "import_micrographs",
            params={
                "blob_paths": blob_pattern,
                "psize_A": cfg.pixel_size_A,
                "accel_kv": cfg.accel_kv,
                "cs_mm": cfg.cs_mm,
                "total_dose_e_per_A2": resolved_dose,
            },
        )
        import_micrographs_job.queue(cfg.lane)

        ctf_estimation_job = workspace.create_job(
            "ctf_estimation",
            connections={"exposures": (import_micrographs_job.uid, "imported_micrographs")},
            params=cfg.ctf_estimation.model_dump(),
        )
        ctf_estimation_job.queue(cfg.lane)


        # For a single particle diameter we can extract directly. For
        # multiple diameters, particles are topaz-extracted per diameter;
        # merging/dedup across diameters before extraction is not yet
        # wired up here (mirrors the original pipeline).
        if len(cfg.particle_diameters_A) == 1:
            topaz_job = workspace.create_job(
                "topaz_extract",
                connections={"micrographs": (ctf_estimation_job.uid, "exposures_accepted")},
                params={
                    "exec_path": cfg.topaz.exec_path,
                    "pretrained": cfg.topaz.pretrained,
                    "par_diam": cfg.particle_diameters_A[0],
                    "threshold": cfg.topaz.threshold,
                },
            )
            topaz_job.queue(cfg.lane)
            topaz_job.wait_for_done()

            extract_job = workspace.create_job(
                "extract_micrographs_cpu_parallel",
                connections={
                    "micrographs": (topaz_job.uid, "micrographs"),
                    "particles": (topaz_job.uid, "particles"),
                },
                params={"box_size_pix": cfg.box_size_pix},
            )
            extract_job.queue(cfg.lane)
        else:
            for par_diam in cfg.particle_diameters_A:
                topaz_job = workspace.create_job(
                    "topaz_extract",
                    connections={"micrographs": (ctf_estimation_job.uid, "exposures_accepted")},
                    params={
                        "exec_path": cfg.topaz.exec_path,
                        "pretrained": cfg.topaz.pretrained,
                        "par_diam": par_diam,
                        "threshold": cfg.topaz.threshold,
                    },
                )
                topaz_job.queue(cfg.lane)
                topaz_job.wait_for_done()

        print(f"Queued CryoSPARC import pipeline for {data_dir} (dose={resolved_dose})")


def _load_mrc_dirs_and_dose_map(
    mrc_dirs_file: Path, mrc_dose_map_file: Path
) -> tuple[list[Path], dict[str, float]]:
    mrc_dirs = [Path(line) for line in mrc_dirs_file.read_text().splitlines() if line.strip()]

    dose_map: dict[str, float] = {}
    if mrc_dose_map_file.exists():
        raw_map = json.loads(mrc_dose_map_file.read_text())
        for key, value in raw_map.items():
            try:
                dose_map[str(Path(key))] = float(value)
            except (TypeError, ValueError):
                continue
    return mrc_dirs, dose_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CryoSPARC micrograph import and refinement pipeline."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "--mrc-dirs-file", type=Path, default=None, help="Override mrc-dirs-file from the config."
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    microscope = resolve_microscope_spec(raw["microscope"]) if "microscope" in raw else None
    cfg = CryoSPARCImportConfig.model_validate(raw["cryosparc"])

    mrc_dirs_file = args.mrc_dirs_file or cfg.mrc_dirs_file
    mrc_dose_map_file = cfg.resolved_mrc_dose_map_file()
    mrc_dirs, dose_map = _load_mrc_dirs_and_dose_map(mrc_dirs_file, mrc_dose_map_file)

    run_import(cfg, microscope, mrc_dirs, dose_map)


if __name__ == "__main__":
    main()
