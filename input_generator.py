"""Generate a TEM-simulator input file from a SimulationConfig.

All microscope, detector, sample and mixture parameters come from a YAML
config file (see config.py and configs/example_run.yaml) instead of being
hardcoded here or scattered across CLI flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import SimulationConfig, load_simulation_config


def _section(name: str, params: dict[str, object]) -> str:
    lines = [f"=== {name} ==="]
    for key, value in params.items():
        lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _write_input_file(sections: list[tuple[str, dict[str, object]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_section(name, params) for name, params in sections)
    out_path.write_text(text, encoding="ascii")


def generate_input(config: SimulationConfig) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = config.run_id or str(np.random.randint(1000, 99999))
    log_file = output_dir / f"log_{run_id}.txt"
    image_file_out = output_dir / f"{run_id}.mrc"
    input_file = config.input_file or output_dir / f"input_{run_id}.txt"

    # 1) Particle definitions, one per species. Each species declares its
    #    source (map or pdb) explicitly in config -- no guessing from
    #    whichever file happens to exist on disk.
    particle_defs: list[tuple[str, dict[str, object]]] = []
    for species in config.mixture.species:
        resolved_path = species.resolved_path()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Structure file for species '{species.name}' "
                f"(source={species.source.value}) not found: {resolved_path}"
            )
        particle_defs.append((species.name, species.particle_params()))

    if config.mixture.noise.enabled:
        particle_defs.append(("background", config.mixture.noise.background_particle_params()))
        particle_defs.append(("crystal", config.mixture.noise.crystal_particle_params()))

    # 2) Particle sets: where/how many of each particle type appear.
    particlesets: list[tuple[str, dict[str, object]]] = [
        (
            species.name,
            {
                "particle_coords": "random",
                "where": "volume",
                "particle_conc": species.concentration,
                "coord_file_out": output_dir / f"coord_{run_id}_{species.name}.txt",
            },
        )
        for species in config.mixture.species
    ]
    if config.mixture.noise.enabled:
        particlesets.append(
            (
                "background",
                {
                    "particle_coords": "random",
                    "occupancy": str(config.mixture.noise.background_occupancy),
                    "where": "volume",
                },
            )
        )
        particlesets.append(
            (
                "crystal",
                {
                    "particle_coords": "random",
                    "where": "top",
                    "num_particles": config.mixture.noise.crystal_num_particles,
                },
            )
        )

    # 3) Microscope config with run-specific overrides (dose, defocus).
    electronbeam = config.microscope.electronbeam.model_dump()
    electronbeam["total_dose"] = config.dose * 100  # e/A^2 -> e/nm^2

    optics = config.microscope.optics.model_dump()
    optics["defocus_nominal"] = config.defocus

    detector = config.microscope.detector.model_dump()
    detector["image_file_out"] = image_file_out

    # 4) Assemble input sections in readable order.
    sections: list[tuple[str, dict[str, object]]] = []
    sections.append(("simulation", {"generate_micrographs": "yes"}))
    sections.append(("simulation", {"log_file": log_file}))

    sections.append(("sample", config.sample.model_dump()))

    for name, params in particle_defs:
        sections.append((f"particle {name}", params))

    for particle_type, params in particlesets:
        sections.append(("particleset", {"particle_type": particle_type, **params}))

    sections.append(
        (
            "geometry",
            {
                "gen_tilt_data": "yes",
                "theta_start": 0,
                "theta_incr": 0,
                "ntilts": 1,
                "tilt_mode": "single_particle",
                "geom_errors": "none",
            },
        )
    )

    sections.append(("electronbeam", electronbeam))
    sections.append(("optics", optics))
    sections.append(("detector", detector))

    _write_input_file(sections, input_file)
    return input_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TEM-simulator input file from a YAML config."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a simulation YAML config.")
    parser.add_argument("--run-id", default=None, help="Override run id from the config.")
    parser.add_argument("--dose", type=float, default=None, help="Override dose (e/A^2).")
    parser.add_argument("--defocus", type=float, default=None, help="Override defocus (micrometers).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--input-file", type=Path, default=None, help="Override output input-file path.")
    args = parser.parse_args()

    config = load_simulation_config(args.config)

    overrides = {
        key: value
        for key, value in (
            ("run_id", args.run_id),
            ("dose", args.dose),
            ("defocus", args.defocus),
            ("output_dir", args.output_dir),
            ("input_file", args.input_file),
        )
        if value is not None
    }
    if overrides:
        config = config.model_copy(update=overrides)

    try:
        path = generate_input(config)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    else:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
