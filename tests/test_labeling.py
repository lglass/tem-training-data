"""Unit tests for labeling.label_picks, plus a gated acceptance test against
the real J58 dataset (skipped unless /mnt/data is mounted)."""

from pathlib import Path

import numpy as np
import pytest

import labeling
import pick_distances


def _gt(species, dists_from_origin):
    """Ground truth points on the x-axis at the given distances from the origin."""
    n = len(species)
    return labeling.GroundTruth(
        x_frac=np.array(dists_from_origin, dtype=float),
        y_frac=np.zeros(n),
        species=np.array(species, dtype=object),
        rot=np.arange(n, dtype=float),
        tilt=np.arange(n, dtype=float),
        psi=np.arange(n, dtype=float),
    )


def _label(gt, match_radius_A=30.0, ambiguity_radius_A=60.0):
    result = labeling.label_picks(
        np.array([0.0]), np.array([0.0]), gt,
        a_per_frac=1.0, match_radius_A=match_radius_A, ambiguity_radius_A=ambiguity_radius_A,
    )
    return result.label[0], result


def test_no_ground_truth_is_nogt():
    label, result = _label(None)
    assert label == labeling.NO_GT
    assert result.species_nearest[0] == labeling.NO_GT
    assert np.isnan(result.dist_A[0])


def test_empty_when_nothing_within_radius():
    gt = _gt(["hexamer"], [50.0])
    label, result = _label(gt, match_radius_A=30.0)
    assert label == labeling.EMPTY
    # species_nearest/dist_A still populated for diagnostics even on a miss
    assert result.species_nearest[0] == "hexamer"
    assert result.dist_A[0] == 50.0


def test_clear_hexamer():
    gt = _gt(["hexamer", "dimer"], [5.0, 500.0])
    label, _ = _label(gt)
    assert label == "hexamer"


def test_clear_dimer():
    gt = _gt(["hexamer", "dimer"], [500.0, 5.0])
    label, _ = _label(gt)
    assert label == "dimer"


def test_ambiguity_radius_sweep():
    gt = _gt(["hexamer", "dimer"], [10.0, 50.0])
    label_60, _ = _label(gt, match_radius_A=30.0, ambiguity_radius_A=60.0)
    assert label_60 == labeling.AMBIGUOUS
    label_40, _ = _label(gt, match_radius_A=30.0, ambiguity_radius_A=40.0)
    assert label_40 == "hexamer"


def test_both_species_within_match_radius_is_ambiguous():
    gt = _gt(["hexamer", "dimer"], [5.0, 8.0])
    label, _ = _label(gt, match_radius_A=30.0, ambiguity_radius_A=60.0)
    assert label == labeling.AMBIGUOUS


def test_single_species_dist_other_is_inf_never_ambiguous():
    gt = _gt(["hexamer", "hexamer"], [5.0, 20.0])
    label, result = _label(gt, match_radius_A=30.0, ambiguity_radius_A=10000.0)
    assert np.isinf(result.dist_other_A[0])
    assert label == "hexamer"


# --- acceptance test against the real J58 dataset, skipped unless /mnt/data is mounted ---

CS_PATH = Path("/mnt/data/CS-lonp/J58/extracted_particles.cs")
STAR_ROOT = Path("/mnt/data/lonp")


@pytest.mark.skipif(not CS_PATH.exists() or not STAR_ROOT.exists(), reason="requires /mnt/data")
@pytest.mark.parametrize(
    "ambiguity_radius_A,expected",
    [
        (29.9008, {"hexamer": 79463, "dimer": 31578, "Ambiguous": 4144, "Empty": 14457}),
        (60.0, {"hexamer": 71377, "dimer": 30568, "Ambiguous": 13240, "Empty": 14457}),
    ],
)
def test_j58_acceptance_counts(ambiguity_radius_A, expected):
    picks_df, mic_ids, a_per_frac, star_idx = pick_distances.load_job(CS_PATH, STAR_ROOT, mic_dim=4096)
    labeled = pick_distances.label_all(
        picks_df, mic_ids, a_per_frac, star_idx,
        match_radius_A=29.9008, ambiguity_radius_A=ambiguity_radius_A,
    )
    counts = labeled["label"].value_counts().to_dict()
    assert sum(counts.values()) == 129642
    for k, v in expected.items():
        assert counts[k] == v
