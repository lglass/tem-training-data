# Getting started

## Install the Simulator
Get simulation software and install

```
wget https://gigenet.dl.sourceforge.net/project/tem-simulator/TEM-simulator_1.3.zip?viasf=1
```

(Alternatively visit [http://tem-simulator.sourceforge.net](http://tem-simulator.sourceforge.net))

The fftw3 location on Linux should be 

```
-I /usr/include -L /usr/lib/
```

## Python environment

Choose your preferred environment manager below to create the `protein-mixtures` environment and install the required dependencies.

### Option 1: uv

```bash
# Create the virtual environment
uv venv protein-mixtures

# Activate the environment
source protein-mixtures/bin/activate  # On Windows: protein-mixtures\Scripts\activate

# Install dependencies
uv pip install mrcfile starfile scipy cryosparc-tools python-dotenv tifffile torch torchvision pytorch-ignite PyYAML
```

### Option 2: conda
```bash
conda create -n protein-mixtures python=3 -y
conda activate protein-mixtures

# Install dependencies
pip install mrcfile starfile scipy cryosparc-tools python-dotenv tifffile torch torchvision pytorch-ignite PyYAML
```

# Running TEM-simulator / CryoSPARC pipeline

Five scripts sharing one YAML-based configuration model (`config.py`):

- `input_generator.py` — writes a single TEM-simulator input file.
- `run_sim_parallel.py` — runs a batch of simulations in parallel, then
  optionally imports the results into CryoSPARC.
- `cryosparc_run.py` — imports micrographs into CryoSPARC and runs
  import → CTF estimation → topaz extraction. Usable standalone or
  chained automatically from `run_sim_parallel.py`.
- `merge_particles.py` — merges particle picks from multiple CryoSPARC jobs
  (e.g. topaz extractions run at different particle diameters), deduplicating
  by pick location. Run by hand after `cryosparc_run.py`'s picking jobs when
  more than one `particle_diameters_A` value was used.
- `training_data.py` — collects extracted particles back out of CryoSPARC,
  labels each one against the TEM-simulator ground truth (the `.star` files
  `txt2star.py` produces) by pick location, and writes a class-balanced CSV
  for training a classifier. Labels are the matched species, `Empty` (no
  ground-truth particle within `match_radius_A`, in Angstrom), `Ambiguous`
  (a particle of another species also within `ambiguity_radius_A`), or
  `NoGT` (no ground truth at all for that micrograph) — see `labeling.py`,
  shared with the offline diagnostic `pick_distances.py`. `--dry-run` prints
  label counts without writing the CSV.

See `configs/example_run.yaml`, `configs/lysate_batch.yaml`,
`configs/cryosparc_only.yaml`, `configs/merge_particles.yaml` and
`configs/training_data.yaml` for complete examples.

## Microscope presets

`microscopes.yaml` holds named microscope+detector presets. Reference one by name, or override part of it inline:

```yaml
microscope: glacios
# or:
microscope: {preset: glacios, optics: {cs: 2.0}}
# or a fully inline definition with electronbeam/optics/detector keys
```

The pixel size, accelerating voltage and spherical aberration used for
CryoSPARC import are derived from this microscope config.

## Protein mixture

```yaml
mixture:
  species:
    - name: 4AAR
      source: map      # looks for structures/4AAR_map.mrc unless `path:` is given
      concentration: 400
    - name: 6PXM
      source: pdb       # looks for structures/6PXM.pdb unless `path:` is given
      concentration: 400
  noise:
    enabled: true        # optional fixed background particle + ice crystal
```

## Single run

```
python input_generator.py --config configs/example_run.yaml
```

`--run-id`, `--dose`, `--defocus`, `--output-dir`, `--input-file` may be
passed to override the loaded config for one-off manual runs.

## Parallel batch

```
python run_sim_parallel.py --config configs/lysate_batch.yaml
```

- The mixture, microscope, sample geometry, dose values and defocus source
  all come from the config (`dose_values`, `num_micrographs`, `defocus:`,
  `max_workers`, `output_dir`, `input_dir`).
- `defocus:` supports `{type: npy, path: ...}` (draw from an empirical sample), `{type: range, min, max}` (uniform), or `{type: list, values: [...]}`. Set `seed:` for reproducible sampling.
- Each worker calls `input_generator.generate_input()` in-process; only the `TEM-simulator`
  binary itself runs as a subprocess.
- If the config has a `cryosparc:` section, results are imported into
  CryoSPARC automatically once the batch finishes, using the dose values
  collected directly from job results. Pass `--skip-cryosparc` to opt out for one run.

## CryoSPARC import (standalone)

```
python cryosparc_run.py --config configs/cryosparc_only.yaml
```

Useful for re-importing an existing `mrc_dirs.txt` without re-running the
simulation. `microscope:` is optional here; if omitted, set `pixel_size_A`,
`accel_kv` and `cs_mm` explicitly under `cryosparc:`.

Connection credentials are not stored in the YAML — they're loaded from
an env file (`cryosparc.connection.env_file`, default `~/.env`) containing
`CRYOSPARC_LICENSE_ID`, `CRYOSPARC_EMAIL`, `CRYOSPARC_PASSWORD`,
`CRYOSPARC_HOST`, `CRYOSPARC_PORT`. The other CryoSPARC-facing scripts below
(`merge_particles.py`, `training_data.py`) load credentials the same way,
via their own `connection:` section.

## Merging particle picks

```
python merge_particles.py --config configs/merge_particles.yaml
```

Deduplicates particle picks from the CryoSPARC jobs listed in `job_ids`
(priority order — the first job a given pick location is seen in wins),
and saves the merged set back into the workspace as an external result.
Pass job ids on the command line to override `job_ids` from the config,
e.g. `python merge_particles.py --config configs/merge_particles.yaml J10 J11 J12`.

Needed when `cryosparc_run.py` was run with more than one
`particle_diameters_A` value — each diameter gets its own topaz/extraction
jobs, and picks aren't merged across diameters automatically.

## Collecting training data

```
python training_data.py --config configs/training_data.yaml
```

Walks every `particle_job_type` job (default `downsample_particles`) in
`project`/`workspace`, matches each extracted particle to the TEM-simulator
ground truth by pick location (within `match_radius`, fractional
coordinates), and labels it via `label_map` (unmatched picks are `"Empty"` →
`0`). `label_map` isn't derived automatically — keep it in sync by hand with
whatever mixture(s) were simulated into the project. Output is capped at
`sample_size` rows per class for balance and written to `output_csv`.


