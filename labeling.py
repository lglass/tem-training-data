"""Shared pick-labelling rule for CryoSPARC picks against simulated ground truth.

Given a pick's fractional position on a micrograph and the ground-truth
particles simulated onto that micrograph (one or more species), labels the
pick as:

  - "NoGT"      no ground truth is available for this micrograph at all
                (e.g. no coord_<mic_id>_*.star file was found).
  - "Empty"     ground truth is available, but no particle of any species is
                within `match_radius_A` of the pick.
  - "Ambiguous" a particle is within `match_radius_A`, but a particle of a
                *different* species is within `ambiguity_radius_A` as well,
                making the choice of nearest species close to arbitrary.
  - <species>   otherwise: the species of the nearest particle.

No CryoSPARC dependency -- this module only needs numpy/scipy, so it is
usable offline (see pick_distances.py) as well as from training_data.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

EMPTY = "Empty"
AMBIGUOUS = "Ambiguous"
NO_GT = "NoGT"


@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth particles simulated onto one micrograph, in fractional coordinates."""

    x_frac: np.ndarray
    y_frac: np.ndarray
    species: np.ndarray  # dtype=object, one per particle
    rot: np.ndarray
    tilt: np.ndarray
    psi: np.ndarray

    def __len__(self) -> int:
        return len(self.x_frac)

    @classmethod
    def from_coords(
        cls,
        x_px: np.ndarray,
        y_px: np.ndarray,
        species: np.ndarray,
        rot: np.ndarray,
        tilt: np.ndarray,
        psi: np.ndarray,
        mic_dim: int,
    ) -> "GroundTruth | None":
        """Build from pixel coordinates on a `mic_dim`-px-square simulated micrograph.

        Bounds-filters to [0, mic_dim] and converts to fractional coordinates.
        Returns None if nothing survives -- callers should treat that exactly
        like "no ground truth found for this micrograph" (NO_GT), not EMPTY.
        """
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        keep = (x_px >= 0) & (x_px <= mic_dim) & (y_px >= 0) & (y_px <= mic_dim)
        if not np.any(keep):
            return None
        return cls(
            x_frac=x_px[keep] / mic_dim,
            y_frac=y_px[keep] / mic_dim,
            species=np.asarray(species, dtype=object)[keep],
            rot=np.asarray(rot, dtype=float)[keep],
            tilt=np.asarray(tilt, dtype=float)[keep],
            psi=np.asarray(psi, dtype=float)[keep],
        )


@dataclass(frozen=True)
class LabelResult:
    label: np.ndarray  # dtype=object: species name | EMPTY | AMBIGUOUS | NO_GT
    species_nearest: np.ndarray  # dtype=object: nearest species regardless of hit, or NO_GT
    dist_A: np.ndarray  # nan if NO_GT
    dist_other_A: np.ndarray  # inf if only one species present on the mic; nan if NO_GT
    rot: np.ndarray  # of the nearest particle; nan if NO_GT
    tilt: np.ndarray
    psi: np.ndarray
    dx_A: np.ndarray  # pick -> nearest particle offset; nan if NO_GT
    dy_A: np.ndarray


def _no_gt_result(n: int) -> LabelResult:
    nans = np.full(n, np.nan)
    return LabelResult(
        label=np.full(n, NO_GT, dtype=object),
        species_nearest=np.full(n, NO_GT, dtype=object),
        dist_A=nans.copy(),
        dist_other_A=nans.copy(),
        rot=nans.copy(),
        tilt=nans.copy(),
        psi=nans.copy(),
        dx_A=nans.copy(),
        dy_A=nans.copy(),
    )


def label_picks(
    pick_x_frac: np.ndarray,
    pick_y_frac: np.ndarray,
    gt: GroundTruth | None,
    a_per_frac: float,
    match_radius_A: float,
    ambiguity_radius_A: float,
) -> LabelResult:
    """Label picks on one micrograph against that micrograph's ground truth.

    `a_per_frac` is Angstrom per unit of fractional coordinate (see
    `angstrom_per_frac`); it may vary per micrograph but is scalar within one
    call, since all picks passed in one call belong to the same micrograph.
    """
    pick_x_frac = np.asarray(pick_x_frac, dtype=float)
    pick_y_frac = np.asarray(pick_y_frac, dtype=float)
    n = len(pick_x_frac)

    if gt is None or len(gt) == 0:
        return _no_gt_result(n)

    pick_xy = np.stack([pick_x_frac, pick_y_frac], axis=1)
    gt_xy = np.stack([gt.x_frac, gt.y_frac], axis=1)

    species_list = sorted(np.unique(gt.species))
    sp_arr = gt.species
    dists, idxs = {}, {}
    for sp in species_list:
        sp_indices = np.where(sp_arr == sp)[0]
        dd, ii = cKDTree(gt_xy[sp_indices]).query(pick_xy, k=1)
        dists[sp], idxs[sp] = dd, sp_indices[ii]

    D = np.stack([dists[sp] for sp in species_list], axis=1)
    rows = np.arange(n)
    nearest_col = D.argmin(axis=1)
    d1 = D[rows, nearest_col]

    D_other = D.copy()
    D_other[rows, nearest_col] = np.inf
    d_other = D_other.min(axis=1)  # inf if only one species is present

    j = np.array([idxs[species_list[c]][r] for r, c in enumerate(nearest_col)])
    species_nearest = np.array(species_list, dtype=object)[nearest_col]

    dist_A = d1 * a_per_frac
    dist_other_A = d_other * a_per_frac

    hit = dist_A < match_radius_A
    ambiguous = hit & (dist_other_A < ambiguity_radius_A)
    label = np.where(~hit, EMPTY, np.where(ambiguous, AMBIGUOUS, species_nearest)).astype(object)

    dx_A = (gt.x_frac[j] - pick_x_frac) * a_per_frac
    dy_A = (gt.y_frac[j] - pick_y_frac) * a_per_frac

    return LabelResult(
        label=label,
        species_nearest=species_nearest,
        dist_A=dist_A,
        dist_other_A=dist_other_A,
        rot=gt.rot[j],
        tilt=gt.tilt[j],
        psi=gt.psi[j],
        dx_A=dx_A,
        dy_A=dy_A,
    )


def angstrom_per_frac(micrograph_shape: np.ndarray, micrograph_psize_A: np.ndarray) -> np.ndarray:
    """Angstrom per unit of fractional coordinate, per pick.

    `micrograph_shape` is (n, 2) px [rows, cols]; `micrograph_psize_A` is (n,)
    Angstrom/px. Raises ValueError if any micrograph is non-square, since a
    fractional distance is only isotropic (and thus convertible by a single
    scalar) for a square micrograph.
    """
    shape = np.asarray(micrograph_shape)
    psize = np.asarray(micrograph_psize_A, dtype=float)
    if not np.all(shape[:, 0] == shape[:, 1]):
        raise ValueError(
            "non-square micrographs: fractional distance is not isotropic; "
            "angstrom_per_frac needs extending to handle this"
        )
    return shape[:, 1].astype(float) * psize
