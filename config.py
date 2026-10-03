"""Shared configuration model for the TEM-simulator / CryoSPARC pipeline.

This module is the single source of truth for every parameter used by
``input_generator.py``, ``run_sim_parallel.py`` and ``cryosparc_run.py``. All
three scripts load one YAML configuration file into these pydantic models
instead of hardcoding presets, mixtures or credentials in Python.

Layout of a config file:

    microscope: <preset name> | {preset: <name>, <overrides>} | <full inline def>
    sample: {diameter, thickness_center, thickness_edge}
    mixture: {name, species: [...], noise: {...}}
    ... plus script-specific fields (see SimulationConfig / ParallelSimulationConfig)

Microscope presets live in ``microscopes.yaml`` next to this file and are
resolved automatically whenever a config references one by name.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

MICROSCOPES_FILE = Path(__file__).parent / "microscopes.yaml"


class _StrictModel(BaseModel):
    """Base for all config models: reject unknown keys instead of silently
    ignoring them (e.g. a mistyped or wrong-script field in a YAML file)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Microscope / detector configuration
# --------------------------------------------------------------------------


class ElectronBeamConfig(_StrictModel):
    acc_voltage: float
    energy_spread: float
    gen_dose: Literal["yes", "no"] = "yes"


class OpticsConfig(_StrictModel):
    magnification: float
    cs: float
    cc: float
    aperture: float
    focal_length: float
    cond_ap_angle: float
    gen_defocus: Literal["yes", "no"] = "yes"


class DetectorConfig(_StrictModel):
    det_pix_x: int
    det_pix_y: int
    pixel_size: float  # micrometers, physical detector pixel pitch
    gain: float
    use_quantization: Literal["yes", "no"] = "yes"
    dqe: float
    mtf_a: float
    mtf_b: float
    mtf_c: float
    mtf_alpha: int
    mtf_beta: int
    mtf_p: int
    mtf_q: int


class MicroscopeConfig(_StrictModel):
    name: str
    electronbeam: ElectronBeamConfig
    optics: OpticsConfig
    detector: DetectorConfig

    @property
    def pixel_size_angstrom(self) -> float:
        """Effective pixel size at the sample, in Angstrom.

        Derived from the detector pitch and magnification rather than
        hardcoded, so CryoSPARC import always matches the microscope that
        actually generated the micrographs.
        """
        return self.detector.pixel_size * 1e4 / self.optics.magnification


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_microscope_presets(path: Path = MICROSCOPES_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Microscope presets file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_microscope(spec: str | dict, presets_path: Path = MICROSCOPES_FILE) -> MicroscopeConfig:
    """Build a MicroscopeConfig from a preset name, an overrides dict, or both.

    ``spec`` may be:
      - a bare preset name, e.g. "glacios"
      - {"preset": "glacios", "optics": {"cs": 2.0}} to override part of a preset
      - a fully inline definition with electronbeam/optics/detector keys
    """
    if isinstance(spec, str):
        spec = {"preset": spec}
    spec = dict(spec)
    preset_name = spec.pop("preset", None)

    if preset_name is not None:
        presets = _load_microscope_presets(presets_path)
        if preset_name not in presets:
            raise ValueError(
                f"Unknown microscope preset {preset_name!r}. "
                f"Available presets: {sorted(presets)}"
            )
        merged = _deep_merge(presets[preset_name], spec)
        merged.setdefault("name", preset_name)
    else:
        merged = spec
        merged.setdefault("name", "custom")

    return MicroscopeConfig.model_validate(merged)


def resolve_microscope_spec(value: object) -> MicroscopeConfig:
    if isinstance(value, MicroscopeConfig):
        return value
    if isinstance(value, (str, dict)):
        return load_microscope(value)
    raise TypeError(
        f"microscope must be a preset name, a mapping, or a MicroscopeConfig; got {type(value)}"
    )


MicroscopeField = Annotated[MicroscopeConfig, BeforeValidator(resolve_microscope_spec)]


# --------------------------------------------------------------------------
# Sample geometry
# --------------------------------------------------------------------------


class SampleGeometryConfig(_StrictModel):
    diameter: float = 910.0
    thickness_center: float = 80.0
    thickness_edge: float = 85.0


# --------------------------------------------------------------------------
# Protein mixture: species (map or pdb source) + concentration, plus noise
# --------------------------------------------------------------------------


class SpeciesSource(str, Enum):
    MAP = "map"
    PDB = "pdb"


class SpeciesConfig(_StrictModel):
    name: str
    source: SpeciesSource
    concentration: float = Field(ge=0)
    path: Path | None = None  # defaults to structures/{name}_map.mrc or structures/{name}.pdb
    voxel_size: float = 0.1  # only used when source == pdb
    famp: float = 0.07
    use_imag_pot: Literal["yes", "no"] = "no"

    def resolved_path(self) -> Path:
        if self.path is not None:
            return self.path
        if self.source == SpeciesSource.MAP:
            return Path(f"structures/{self.name}_map.mrc")
        return Path(f"structures/{self.name}.pdb")

    def particle_params(self) -> dict[str, object]:
        path = self.resolved_path()
        if self.source == SpeciesSource.MAP:
            return {
                "source": "map",
                "map_file_re_in": path,
                "use_imag_pot": self.use_imag_pot,
                "famp": self.famp,
            }
        return {
            "source": "pdb",
            "pdb_file_in": path,
            "voxel_size": self.voxel_size,
            "map_file_re_out": Path(f"structures/{self.name}_map.mrc"),
            "use_imag_pot": self.use_imag_pot,
            "famp": self.famp,
        }


class NoiseConfig(_StrictModel):
    """Fixed background noise particle + surface ice crystal, both optional."""

    enabled: bool = False
    famp: float = 0.07
    background_occupancy: float = 0.25
    background_smoothness: float = 0.5
    background_contrast_re: float = 1.5
    background_voxel_size: float = 0.1
    background_size_px: tuple[int, int, int] = (20, 20, 20)
    crystal_map_path: Path = Path("structures/crystal_map.mrc")
    crystal_num_particles: int = 5

    def background_particle_params(self) -> dict[str, object]:
        nx, ny, nz = self.background_size_px
        return {
            "source": "random",
            "smoothness": self.background_smoothness,
            "contrast_re": self.background_contrast_re,
            "make_positive": "no",
            "voxel_size": self.background_voxel_size,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "use_imag_pot": "no",
            "famp": self.famp,
        }

    def crystal_particle_params(self) -> dict[str, object]:
        return {
            "source": "map",
            "map_file_re_in": self.crystal_map_path,
            "use_imag_pot": "no",
            "famp": self.famp,
        }


class MixtureConfig(_StrictModel):
    name: str = "mixture"
    species: list[SpeciesConfig]
    noise: NoiseConfig = Field(default_factory=NoiseConfig)

    @field_validator("species")
    @classmethod
    def _validate_species(cls, species: list[SpeciesConfig]) -> list[SpeciesConfig]:
        if not species:
            raise ValueError("mixture must contain at least one species")
        names = [s.name for s in species]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate species names in mixture: {names}")
        return species


# --------------------------------------------------------------------------
# Single-run simulation config (input_generator.py)
# --------------------------------------------------------------------------


class SimulationConfig(_StrictModel):
    microscope: MicroscopeField
    sample: SampleGeometryConfig = Field(default_factory=SampleGeometryConfig)
    mixture: MixtureConfig
    dose: float = 30.0  # e/A^2
    defocus: float = 1.0  # micrometers
    output_dir: Path = Path("data")
    run_id: str | None = None
    input_file: Path | None = None


def load_simulation_config(path: str | Path) -> SimulationConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return SimulationConfig.model_validate(raw)


# --------------------------------------------------------------------------
# Defocus sampling (for batch/parallel runs)
# --------------------------------------------------------------------------


class DefocusSourceType(str, Enum):
    RANGE = "range"
    LIST = "list"
    NPY = "npy"


class DefocusSource(_StrictModel):
    type: DefocusSourceType
    min: float | None = None
    max: float | None = None
    values: list[float] | None = None
    path: Path | None = None

    @model_validator(mode="after")
    def _validate_fields_for_type(self) -> "DefocusSource":
        if self.type == DefocusSourceType.RANGE and (self.min is None or self.max is None):
            raise ValueError("defocus type 'range' requires 'min' and 'max'")
        if self.type == DefocusSourceType.LIST and not self.values:
            raise ValueError("defocus type 'list' requires non-empty 'values'")
        if self.type == DefocusSourceType.NPY and self.path is None:
            raise ValueError("defocus type 'npy' requires 'path'")
        return self

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.type == DefocusSourceType.RANGE:
            return rng.uniform(self.min, self.max, size=n)
        if self.type == DefocusSourceType.LIST:
            return rng.choice(np.array(self.values, dtype=float), size=n)
        if self.type == DefocusSourceType.NPY:
            pool = np.load(self.path)
            return rng.choice(pool, size=n)
        raise AssertionError(f"unhandled defocus source type: {self.type}")


# --------------------------------------------------------------------------
# CryoSPARC import configuration (cryosparc_run.py)
# --------------------------------------------------------------------------


class CryoSPARCConnectionConfig(_StrictModel):
    env_file: Path = Field(default_factory=lambda: Path.home() / ".env")

    def load(self) -> dict[str, str]:
        load_dotenv(self.env_file.expanduser())
        keys = [
            "CRYOSPARC_LICENSE_ID",
            "CRYOSPARC_EMAIL",
            "CRYOSPARC_PASSWORD",
            "CRYOSPARC_HOST",
            "CRYOSPARC_PORT",
        ]
        values = {key: os.getenv(key) for key in keys}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required CryoSPARC environment variables in {self.env_file}: {missing}"
            )
        return values


class CryoSPARCJobTargetConfig(_StrictModel):
    """Shared connection/project/workspace/lane fields for anything that
    talks to a CryoSPARC project (import pipeline, merge/dedup, training
    data collection)."""

    connection: CryoSPARCConnectionConfig = Field(default_factory=CryoSPARCConnectionConfig)
    project: str
    workspace: str
    lane: str = "default"


class CTFEstimationConfig(_StrictModel):
    astigmatism_known: bool = True
    spectrum_dim: int = 1024
    min_res_A: float = 30
    max_res_A: float = 5
    min_df_A: float = 10000
    max_df_A: float = 40000
    step_df_A: float = 100
    known_astigmatism_A: float = 0
    slow: bool = False


class TopazConfig(_StrictModel):
    exec_path: str = "/home/cryosparcuser/topaz.sh"
    pretrained: str = "ResNet16 (32 units)"
    threshold: float = 1.0


class CryoSPARCImportConfig(CryoSPARCJobTargetConfig):
    # Derived automatically from the microscope config when left unset; set
    # explicitly only when importing micrographs that came from elsewhere.
    pixel_size_A: float | None = None
    accel_kv: float | None = None
    cs_mm: float | None = None

    dose_fallback: float = 30.0
    particle_diameters_A: list[float] = Field(default_factory=lambda: [130.0, 270.0])
    box_size_pix: int = 384
    ctf_estimation: CTFEstimationConfig = Field(default_factory=CTFEstimationConfig)
    ctf_fit_threshold_A: tuple[float, float] = (0, 16)
    topaz: TopazConfig = Field(default_factory=TopazConfig)

    mrc_dirs_file: Path = Path("mrc_dirs.txt")
    mrc_dose_map_file: Path | None = None

    def resolved_mrc_dose_map_file(self) -> Path:
        if self.mrc_dose_map_file is not None:
            return self.mrc_dose_map_file
        return self.mrc_dirs_file.with_name(f"{self.mrc_dirs_file.stem}_dose_map.json")

    def with_microscope_derived_fields(self, microscope: MicroscopeConfig) -> "CryoSPARCImportConfig":
        """Fill pixel_size_A/accel_kv/cs_mm from the microscope config if unset."""
        data = self.model_dump()
        if data["pixel_size_A"] is None:
            data["pixel_size_A"] = microscope.pixel_size_angstrom
        if data["accel_kv"] is None:
            data["accel_kv"] = microscope.electronbeam.acc_voltage
        if data["cs_mm"] is None:
            data["cs_mm"] = microscope.optics.cs
        return CryoSPARCImportConfig.model_validate(data)


# --------------------------------------------------------------------------
# Particle merge/dedup configuration (merge_particles.py)
# --------------------------------------------------------------------------


class MergeParticlesConfig(CryoSPARCJobTargetConfig):
    # Priority order: the first dataset a given (micrograph, x, y) key is
    # seen in wins the dedup. May be overridden from the CLI.
    job_ids: list[str] = Field(default_factory=list)
    output_name: str = "particles"
    dedup_keys: list[str] = Field(
        default_factory=lambda: [
            "location/micrograph_uid",
            "location/center_x_frac",
            "location/center_y_frac",
        ]
    )


def load_merge_particles_config(path: str | Path) -> MergeParticlesConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return MergeParticlesConfig.model_validate(raw["merge_particles"])


# --------------------------------------------------------------------------
# Training data collection configuration (training_data.py)
# --------------------------------------------------------------------------


class TrainingDataConfig(CryoSPARCJobTargetConfig):
    # CryoSPARC job type whose particle outputs get collected and labeled.
    particle_job_type: str = "downsample_particles"

    # Distance within which an extracted particle is matched to a
    # ground-truth (txt2star.py) pick. Set exactly one of these:
    #   match_radius_A   preferred: Angstrom, portable across pixel sizes.
    #   match_radius     deprecated: a fraction of the micrograph width;
    #                     converted to Angstrom at run time using each job's
    #                     own pixel size, with a warning.
    match_radius: float | None = None
    match_radius_A: float | None = None

    # A matched pick is labeled "Ambiguous" instead of its nearest species if
    # a particle of a *different* species is also within this distance (A).
    ambiguity_radius_A: float = 60.0

    mic_dim: int = 4096

    # Explicit structure-id -> integer class label. Kept in sync by hand with
    # whatever mixture(s) were simulated. "Empty" (no particle within
    # match_radius), "Ambiguous" (another species also within
    # ambiguity_radius_A) and "NoGT" (no ground truth for the micrograph at
    # all) are sentinel labels independent of the simulated species, and are
    # always required.
    label_map: dict[str, int] = Field(
        default_factory=lambda: {"Empty": 0, "Ambiguous": 1, "NoGT": 2}
    )

    output_csv: Path = Path("training.csv")
    sample_size: int = 70000  # per-class cap, for class balance

    @model_validator(mode="after")
    def _validate_match_radius(self) -> "TrainingDataConfig":
        if (self.match_radius is None) == (self.match_radius_A is None):
            raise ValueError(
                "set exactly one of 'match_radius' (deprecated, fraction of micrograph "
                "width) or 'match_radius_A' (preferred, Angstrom)"
            )
        return self

    @field_validator("label_map")
    @classmethod
    def _require_sentinel_labels(cls, label_map: dict[str, int]) -> dict[str, int]:
        missing = {"Empty", "Ambiguous", "NoGT"} - label_map.keys()
        if missing:
            raise ValueError(f"label_map is missing required sentinel labels: {sorted(missing)}")
        return label_map


def load_training_data_config(path: str | Path) -> TrainingDataConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return TrainingDataConfig.model_validate(raw["training_data"])


# --------------------------------------------------------------------------
# Experimental data collection configuration (experiment_training.py)
# --------------------------------------------------------------------------


class ExperimentalParticleJobConfig(_StrictModel):
    """A single CryoSPARC job whose particles are all one known species,
    e.g. a topaz_extract run targeted at one protein on experimental
    (non-simulated) micrographs. Unlike training_data.py's per-particle
    ground-truth matching, every particle pulled from the job gets this
    job's configured label."""

    model_config = ConfigDict(populate_by_name=True)

    job_uid: str = Field(alias="jobid")
    label: int
    name: str


class ExperimentalDataConfig(CryoSPARCJobTargetConfig):
    particles: list[ExperimentalParticleJobConfig]
    output_csv: Path = Path("training_exp.csv")


def load_exp_data_config(path: str | Path) -> ExperimentalDataConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentalDataConfig.model_validate(raw["experiment_data"])


# --------------------------------------------------------------------------
# Batch / parallel run config (run_sim_parallel.py)
# --------------------------------------------------------------------------


class ParallelSimulationConfig(_StrictModel):
    microscope: MicroscopeField
    sample: SampleGeometryConfig = Field(default_factory=SampleGeometryConfig)
    mixture: MixtureConfig

    dose_values: list[float] = Field(default_factory=lambda: [30.0])
    defocus: DefocusSource
    num_micrographs: int = Field(gt=0, default=1)
    seed: int | None = None

    max_workers: int = 8
    output_dir: Path = Path("data")
    input_dir: Path = Path("input_files")
    mrc_dirs_file: Path = Path("mrc_dirs.txt")
    mrc_dose_map_file: Path | None = None

    # If set, run_sim_parallel.py imports the results into CryoSPARC once the
    # batch finishes. Omit this section (or pass --skip-cryosparc) to skip it.
    cryosparc: CryoSPARCImportConfig | None = None

    def resolved_mrc_dose_map_file(self) -> Path:
        if self.mrc_dose_map_file is not None:
            return self.mrc_dose_map_file
        return self.mrc_dirs_file.with_name(f"{self.mrc_dirs_file.stem}_dose_map.json")


def load_parallel_simulation_config(path: str | Path) -> ParallelSimulationConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ParallelSimulationConfig.model_validate(raw)
