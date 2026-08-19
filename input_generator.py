"""Generate configuration file for TEM simulation.

This script creates a configuration file for electron microscopy simulation software,
defining particle types, sample geometry, beam parameters, and detector settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ParticleDef:
    name: str
    params: dict[str, object]


@dataclass(frozen=True)
class ParticleSetDef:
    particle_type: str
    params: dict[str, object]


def _particle_definition_for_structure(structure: str) -> ParticleDef:
    map_re = Path(f"structures/{structure}_map.mrc")
    # map_im = Path(f"structures/{structure}_abs_map.mrc")
    pdb = Path(f"structures/{structure}.pdb")

    if map_re.exists():
        params = {
            "source": "map",
            "map_file_re_in": map_re,
            "use_imag_pot": "no",
            "famp": 0.1,
        }
    else:
        params = {
            "source": "pdb",
            "pdb_file_in": pdb,
            "voxel_size": 0.05,
            "map_file_re_out": map_re,
            # "map_file_im_out": map_im,
            "use_imag_pot": "no",
            "famp": 0.1,
        }
    return ParticleDef(name=structure, params=params)


def _noise_particle_definition() -> ParticleDef:
    return ParticleDef(
        name="marker",
        params={
            "source": "random",
            "smoothness": 0.1,
            "contrast_re": 1,
            "make_positive": "no",
            "voxel_size": 0.2,
            "nx": 10,
            "ny": 10,
            "nz": 10,
            "use_imag_pot": "no",
            "famp": 0.1,
            "map_file_re_out": "structures/marker_map.mrc",
        },
    )


def _ice_crystal_definition() -> ParticleDef:
    return ParticleDef(
        name="crystal",
        params={
            "source": "map",
            "map_file_re_in": "structures/crystal_map.mrc",
            "use_imag_pot": "no",
            "famp": 0.1,
        },
    )


def _section(name: str, params: dict[str, object]) -> str:
    lines = [f"=== {name} ==="]
    for key, value in params.items():
        lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _write_input_file(
    sections: list[tuple[str, dict[str, object]]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_section(name, params) for name, params in sections)
    out_path.write_text(text, encoding="ascii")


def _parse_csv_str(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _parse_csv_float(csv: str) -> list[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


# Microscope parameter set.
MICROSCOPE_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "300kV": {
        "electronbeam": {
            "acc_voltage": 300,
            "energy_spread": 0.3,
            "gen_dose": "yes",
        },
        "optics": {
            "magnification": 191000.0,
            "cs": 2.7,
            "cc": 2.7,
            "aperture": 100,
            "focal_length": 3.4,
            "cond_ap_angle": 0.01,
            "gen_defocus": "yes",
        },
        "detector": {
            "det_pix_x": 4096,
            "det_pix_y": 4096,
            "pixel_size": 14,
            "gain": 1,
            "use_quantization": "yes",
            "dqe": 0.9,
            "mtf_a": 0.5,
            "mtf_b": 0.5,
            "mtf_c": 0.0,
            "mtf_alpha": 7,
            "mtf_beta": 20,
            "mtf_p": 4,
            "mtf_q": 2,
        },
    },
}


def generate_input(args: argparse.Namespace) -> Path:
    structures = _parse_csv_str(args.structures)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or str(np.random.randint(1000, 99999))
    # coord_prefix = output_dir / f"coord_{run_id}"
    log_file = output_dir / f"log_{run_id}.txt"
    image_file_out = output_dir / f"{run_id}.mrc"
    if args.input_file is None:
        input_file = output_dir / f"input_{run_id}.txt"
    else:
        input_file = Path(args.input_file)

    # 2) Build particle definitions.
    particle_defs: list[ParticleDef] = [
        _particle_definition_for_structure(s) for s in structures
    ]
    if args.add_noise:
        particle_defs.append(_noise_particle_definition())
        particle_defs.append(_ice_crystal_definition())

    # 3) Build particlesets.
    particlesets: list[ParticleSetDef] = [
        ParticleSetDef(
            particle_type=s,
            params={
                "particle_coords": "random",
                "where": "volume",
                # "num_particles": 100,
                "particle_conc": 20000,  # particles per cubic micron
                "coord_file_out": output_dir / f"coord_{run_id}_{s}.txt",
            },
        )
        for s in structures
    ]
    if args.add_noise:
        particlesets.append(
            ParticleSetDef(
                particle_type="marker",
                params={
                    "particle_coords": "random",
                    "occupancy": "0.5",
                    "where": "top",
                },
            )
        )
        particlesets.append(
            ParticleSetDef(
                particle_type="crystal",
                params={
                    "particle_coords": "random",
                    "where": "top",
                    "num_particles": 5,
                },
            )
        )

    # 4) Microscope preset with run-specific overrides.
    microscope = MICROSCOPE_PRESETS["300kV"]
    electronbeam = dict(microscope["electronbeam"])
    electronbeam["total_dose"] = args.dose * 100  # e/A^2 -> e/nm^2
    if args.energy_spread is not None:
        electronbeam["energy_spread"] = args.energy_spread

    optics = dict(microscope["optics"])
    optics["defocus_nominal"] = args.defocus
    if args.con_ap_angle is not None:
        optics["cond_ap_angle"] = args.con_ap_angle
    if args.aperture is not None:
        optics["aperture"] = args.aperture

    detector = dict(microscope["detector"])
    detector["image_file_out"] = image_file_out
    if args.dqe is not None:
        detector["dqe"] = args.dqe
    if args.mtf_a is not None:
        detector["mtf_a"] = args.mtf_a
    if args.mtf_b is not None:
        detector["mtf_b"] = args.mtf_b
    if args.mtf_c is not None:
        detector["mtf_c"] = args.mtf_c
    if args.mtf_alpha is not None:
        detector["mtf_alpha"] = args.mtf_alpha
    if args.mtf_beta is not None:
        detector["mtf_beta"] = args.mtf_beta

    # 5) Assemble input sections in readable order.
    sections: list[tuple[str, dict[str, object]]] = []
    sections.append(("simulation", {"generate_micrographs": "yes"}))
    sections.append(("simulation", {"log_file": log_file}))

    sections.append(
        (
            "sample",
            {
                "diameter": args.sample_diameter,
                "thickness_center": args.thickness_center,
                "thickness_edge": args.thickness_edge,
            },
        )
    )

    for pdef in particle_defs:
        sections.append((f"particle {pdef.name}", pdef.params))

    for pset in particlesets:
        pset_params = {"particle_type": pset.particle_type, **pset.params}
        sections.append(
            (
                "particleset",
                pset_params,
            )
        )

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
        description="Generate readable TEM-simulator input files."
    )

    # Filenames / paths
    parser.add_argument(
        "--input-file",
        default=None,
        help="Output TEM-simulator input file (default: <output-dir>/input_<run-id>.txt).",
    )
    parser.add_argument(
        "--output-dir", default="data", help="Directory for generated coordinate files."
    )
    parser.add_argument(
        "--run-id", default=None, help="Run id used in default filenames (e.g. 5715)."
    )

    # Particle definitions / particle sets
    parser.add_argument(
        "--structures", default="4AAR,6PXM", help="Comma-separated structure names."
    )

    parser.add_argument(
        "--add-noise",
        action="store_true",
        help="Add a fixed random marker particle and particleset used as noise.",
    )

    # Sample geometry
    parser.add_argument("--sample-diameter", type=float, default=430.0)
    parser.add_argument("--thickness-center", type=float, default=50.0)
    parser.add_argument("--thickness-edge", type=float, default=55.0)

    # Parameter set (preset) + run-specific overrides.
    parser.add_argument(
        "--defocus", type=float, default=1.75, help="Defocus in micrometers."
    )
    parser.add_argument("--dose", type=float, default=45.0, help="Dose in e/A^2.")
    parser.add_argument(
        "--energy-spread",
        type=float,
        default=None,
        help="Override electron beam energy spread.",
    )
    parser.add_argument(
        "--con-ap-angle",
        type=float,
        default=None,
        help="Override condenser aperture angle (cond_ap_angle).",
    )
    parser.add_argument(
        "--aperture", type=float, default=None, help="Override objective aperture."
    )
    parser.add_argument(
        "--dqe", type=float, default=None, help="Override detector DQE."
    )
    parser.add_argument(
        "--mtf-a", type=float, default=None, help="Override detector mtf_a."
    )
    parser.add_argument(
        "--mtf-b", type=float, default=None, help="Override detector mtf_b."
    )
    parser.add_argument(
        "--mtf-c", type=float, default=None, help="Override detector mtf_c."
    )
    parser.add_argument(
        "--mtf-alpha", type=float, default=None, help="Override detector mtf_alpha."
    )
    parser.add_argument(
        "--mtf-beta", type=float, default=None, help="Override detector mtf_beta."
    )

    args = parser.parse_args()
    generate_input(args)


if __name__ == "__main__":
    main()
