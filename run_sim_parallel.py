#!/usr/bin/env python
"""Run a batch of TEM-simulator jobs in parallel from a single YAML config.

The mixture, microscope, sample geometry, dose values and defocus source all
come from config.ParallelSimulationConfig -- there is no mixture or output
path hardcoded in this file. Each job calls input_generator.generate_input()
directly (in-process) instead of shelling out to a second Python process, and
only the actual TEM-simulator binary is run as a subprocess.

If the config has a `cryosparc:` section, the results are imported into
CryoSPARC automatically once the batch finishes (pass --skip-cryosparc to
opt out for a given run).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from config import ParallelSimulationConfig, SimulationConfig, load_parallel_simulation_config
from input_generator import generate_input

# cryosparc_run pulls in the `cryosparc` package, which isn't needed unless a
# `cryosparc:` section is actually configured -- import it lazily below.


class JobResult(BaseModel):
    status: Literal["success", "error"]
    run_id: str
    micrograph: int
    dose: float
    defocus: float
    output_dir: str | None = None
    error: str | None = None


def run_cmd(cmd: list[str], cwd: Path | None, quiet: bool = False) -> None:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.STDOUT if quiet else None
    subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout, stderr=stderr)


def format_dose_dir(dose: float) -> str:
    return f"dose_{dose:.1f}"


def build_job_configs(batch: ParallelSimulationConfig) -> list[tuple[SimulationConfig, int]]:
    rng = np.random.default_rng(batch.seed)
    n_total = batch.num_micrographs * len(batch.dose_values)
    defocus_values = iter(batch.defocus.sample(rng, n_total))

    jobs: list[tuple[SimulationConfig, int]] = []
    for mic_idx in range(1, batch.num_micrographs + 1):
        for dose in batch.dose_values:
            run_id = uuid.uuid4().hex[:8]
            mrc_dir = batch.output_dir / format_dose_dir(dose)
            config = SimulationConfig(
                microscope=batch.microscope,
                sample=batch.sample,
                mixture=batch.mixture,
                dose=dose,
                defocus=float(next(defocus_values)),
                output_dir=mrc_dir,
                run_id=run_id,
                input_file=batch.input_dir / f"input_{run_id}.txt",
            )
            jobs.append((config, mic_idx))
    return jobs


def run_single_job(config: SimulationConfig, micrograph_idx: int) -> JobResult:
    try:
        input_file = generate_input(config)
        run_cmd(["TEM-simulator", str(input_file)], cwd=Path(".").resolve(), quiet=True)
        return JobResult(
            status="success",
            run_id=config.run_id,
            micrograph=micrograph_idx,
            dose=config.dose,
            defocus=config.defocus,
            output_dir=str(config.output_dir),
        )
    except subprocess.CalledProcessError as error:
        return JobResult(
            status="error",
            run_id=config.run_id,
            micrograph=micrograph_idx,
            dose=config.dose,
            defocus=config.defocus,
            error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - surface any failure as a JobResult
        return JobResult(
            status="error",
            run_id=config.run_id or "unknown",
            micrograph=micrograph_idx,
            dose=config.dose,
            defocus=config.defocus,
            error=f"Unexpected error: {error}",
        )


def run_batch(batch: ParallelSimulationConfig) -> tuple[list[JobResult], dict[str, float]]:
    batch.output_dir.mkdir(parents=True, exist_ok=True)
    batch.input_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_job_configs(batch)
    print(f"Scheduled {len(jobs)} experiments")
    print(f"Using {batch.max_workers} parallel workers")
    print(f"Output directory: {batch.output_dir}")
    print()

    results: list[JobResult] = []
    dose_map: dict[str, float] = {}

    with ProcessPoolExecutor(max_workers=batch.max_workers) as executor:
        futures = {
            executor.submit(run_single_job, config, mic_idx): config for config, mic_idx in jobs
        }

        completed = 0
        start_time = time.time()

        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"✗ [{completed}/{len(jobs)}] Exception: {error}")
                continue

            results.append(result)
            if result.status == "success":
                dose_map[result.output_dir] = result.dose
                print(
                    f"✓ [{completed}/{len(jobs)}] Run {result.run_id} "
                    f"(defocus={result.defocus:.3f}µm, dose={result.dose}e/Å²)"
                )
            else:
                print(
                    f"✗ [{completed}/{len(jobs)}] Run {result.run_id} "
                    f"(defocus={result.defocus:.3f}µm, dose={result.dose}e/Å²) - ERROR"
                )
                print(f"  Error: {result.error or 'Unknown error'}")

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.status == "success")
    failed = len(results) - successful

    print()
    print("=" * 70)
    print("SIMULATION RUN COMPLETE")
    print("=" * 70)
    print(f"Total:        {len(jobs)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Unique output directories: {len(dose_map)}")
    print()

    if dose_map:
        batch.mrc_dirs_file.write_text("".join(f"{d}\n" for d in sorted(dose_map)))
        print(f"Output directories logged to: {batch.mrc_dirs_file}")

        dose_map_file = batch.resolved_mrc_dose_map_file()
        dose_map_file.write_text(json.dumps(dose_map, indent=2, sort_keys=True))
        print(f"Dose map logged to: {dose_map_file}")

    if failed:
        print()
        print("Failed experiments:")
        for result in results:
            if result.status == "error":
                print(f"  - Run {result.run_id}: {result.error or 'Unknown error'}")

    return results, dose_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallelized TEM-simulator runner.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a batch YAML config.")
    parser.add_argument(
        "--skip-cryosparc",
        action="store_true",
        help="Skip CryoSPARC import even if the config has a `cryosparc:` section.",
    )
    args = parser.parse_args()

    batch = load_parallel_simulation_config(args.config)
    results, dose_map = run_batch(batch)

    if batch.cryosparc is None:
        return
    if args.skip_cryosparc:
        print("Skipping CryoSPARC import (--skip-cryosparc).")
        return
    if not dose_map:
        print("No successful runs; skipping CryoSPARC import.")
        return

    print()
    print("Importing results into CryoSPARC...")
    try:
        import cryosparc_run

        cryosparc_run.run_import(
            batch.cryosparc,
            batch.microscope,
            mrc_dirs=[Path(d) for d in sorted(dose_map)],
            dose_map=dose_map,
        )
    except Exception as error:  # noqa: BLE001
        print(f"CryoSPARC import failed: {error}")
        raise


if __name__ == "__main__":
    main()
