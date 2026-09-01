#!/usr/bin/env python

import argparse
import datetime
import json
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple, List

import numpy as np


class ExperimentConfig(NamedTuple):
    name: str
    species: list[tuple[str, float]]
    dose: float
    defocus: float
    micrograph_idx: int
    run_id: str


def run_cmd(cmd: list[str], cwd: Path | None, quiet: bool = False) -> None:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.STDOUT if quiet else None
    subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout, stderr=stderr)


def format_dose_dir(dose: float) -> str:
    return f"dose_{dose:.1f}"


def format_structure_specs(species: list[tuple[str, float]]) -> str:
    return ",".join(f"{name}:{concentration:g}" for name, concentration in species)


def run_single_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    input_dir: Path,
) -> dict:
    try:
        # Generate output directory for this mixture
        base_mrc_dir = output_dir
        mrc_dir = base_mrc_dir / format_dose_dir(config.dose)
        mrc_dir.mkdir(parents=True, exist_ok=True)

        # Unique input and log files for this job
        input_file = input_dir / f"input_{config.run_id}.txt"
        # log_file = mrc_dir / f"log_{config.run_id}.txt"

        input_cmd = [
            "python",
            "input_generator.py",
            "--structures",
            format_structure_specs(config.species),
            "--defocus",
            str(config.defocus),
            "--dose",
            str(config.dose),
            "--output-dir",
            str(mrc_dir),
            "--input-file",
            str(input_file),
            "--run-id",
            config.run_id,
            "--add-noise",
        ]

        project_dir = Path(".").resolve()
        quiet = True
        run_cmd(input_cmd, cwd=project_dir, quiet=quiet)
        run_cmd(["TEM-simulator", str(input_file)], cwd=project_dir, quiet=quiet)

        return {
            "status": "success",
            "run_id": config.run_id,
            "micrograph": config.micrograph_idx,
            "dose": config.dose,
            "defocus": config.defocus,
            "output_dir": str(mrc_dir),
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "run_id": config.run_id,
            "micrograph": config.micrograph_idx,
            "dose": config.dose,
            "defocus": config.defocus,
            "error": str(e),
            "stdout": e.stdout if hasattr(e, "stdout") else "",
            "stderr": e.stderr if hasattr(e, "stderr") else "",
        }
    except Exception as e:
        return {
            "status": "error",
            "run_id": config.run_id,
            "micrograph": config.micrograph_idx,
            "error": f"Unexpected error: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized TEM-simulator runner."
    )
    parser.add_argument(
        "--output-dir",
        default=datetime.date.today().strftime("/mnt/data/lysate/simulated_%Y%m%d"),
        help="Base output directory for simulations.",
    )
    parser.add_argument(
        "--input-dir",
        default="input_files",
        help="Directory to store temporary per-job input files.",
    )
    parser.add_argument(
        "--dose-values",
        default="30",
        help="Comma-separated electron dose values in e/A^2.",
    )
    parser.add_argument(
        "--num-micrographs",
        type=int,
        default=1500,
        help="Number of micrographs to simulate.",
    )
    parser.add_argument(
        "--defocus-experiment",
        type=str,
        default="defocus.npy",
        help="Min,max defocus in micrometers (drawn uniformly per micrograph).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Number of parallel workers."
    )
    parser.add_argument(
        "--mrc-dirs-file",
        default="mrc_dirs.txt",
        help="File to log output directories.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    # Ensure base output and input directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    dose_values = [float(d) for d in args.dose_values.split(",")]
    defocus_exp = np.load(args.defocus_experiment)

    rng = np.random.default_rng()

    mixtures = [
        ("7Q5Q", 768),
        ("7Q5R", 768),
        ("7Q5S", 768),
        ("3JCT", 256),
        ("6ELZ", 256),
        ("6EM5", 256),
        ("1WCM", 90),
        ("4CR2", 90),
        ("4V88_copy1", 90),
        ("5np0", 90),
        ("9pmo", 90),
    ]

    experiments: list[ExperimentConfig] = []
    for mic_idx in range(1, args.num_micrographs + 1):
        for dose in dose_values:
            defocus = rng.choice(defocus_exp)
            run_id = uuid.uuid4().hex[:8]
            experiments.append(
                ExperimentConfig(
                    name="lysate_mixture",
                    species=mixtures,
                    dose=dose,
                    defocus=defocus,
                    micrograph_idx=mic_idx,
                    run_id=run_id,
                )
            )

    print(f"Scheduled {len(experiments)} experiments")
    print(f"Using {args.max_workers} parallel workers")
    print(f"Output directory: {output_dir}")
    print()

    results = []
    mrc_dirs_set = set()

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_config = {
            executor.submit(
                run_single_experiment,
                config,
                output_dir,
                input_dir,
            ): config
            for config in experiments
        }

        completed = 0
        start_time = time.time()

        for future in as_completed(future_to_config):
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    mrc_dirs_set.add(result["output_dir"])
                    print(
                        f"✓ [{completed+1}/{len(experiments)}] "
                        f"Run {result['run_id']} "
                        f"(defocus={result['defocus']:.3f}µm, dose={result['dose']}e/Å²)"
                    )
                else:
                    # Use safe getters for fields that may be missing in error cases
                    defocus_val = result.get("defocus", 0.0)
                    dose_val = result.get("dose", 0.0)
                    print(
                        f"✗ [{completed+1}/{len(experiments)}] "
                        f"Run {result['run_id']} "
                        f"(defocus={defocus_val:.3f}µm, dose={dose_val}e/Å²) - ERROR"
                    )
                    print(f"  Error: {result.get('error', 'Unknown error')}")

                completed += 1

            except Exception as e:
                print(f"✗ [{completed+1}/{len(experiments)}] Exception: {e}")
                completed += 1

    elapsed = time.time() - start_time

    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print()
    print("=" * 70)
    print("SIMULATION RUN COMPLETE")
    print("=" * 70)
    print(f"Total:        {len(experiments)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Unique output directories: {len(mrc_dirs_set)}")
    print()

    if mrc_dirs_set:
        with open(args.mrc_dirs_file, "w") as f:
            for mrc_dir in sorted(mrc_dirs_set):
                f.write(f"{mrc_dir}\n")
        print(f"Output directories logged to: {args.mrc_dirs_file}")

        dose_map_file = Path(args.mrc_dirs_file).with_name(
            f"{Path(args.mrc_dirs_file).stem}_dose_map.json"
        )
        dose_map = {}
        for mrc_dir in sorted(mrc_dirs_set):
            dose_str = Path(mrc_dir).name.replace("dose_", "")
            try:
                dose_map[mrc_dir] = float(dose_str)
            except ValueError:
                continue
        with open(dose_map_file, "w") as f:
            json.dump(dose_map, f, indent=2, sort_keys=True)
        print(f"Dose map logged to: {dose_map_file}")

    if failed > 0:
        print()
        print("Failed experiments:")
        for result in results:
            if result["status"] == "error":
                print(
                    f"  - Run {result['run_id']}: {result.get('error', 'Unknown error')}"
                )


if __name__ == "__main__":
    main()
