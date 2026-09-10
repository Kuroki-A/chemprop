import numpy as np
import pytest

from chemprop.spectra_utils import load_phase_mask, normalize_spectra, roundrobin_sid


def test_normalize_spectra_rejects_rows_without_usable_intensity():
    with pytest.raises(ValueError, match=r"invalid row indices: \[1\]"):
        normalize_spectra([[1.0, 2.0], [None, None]])

    with pytest.raises(ValueError, match="positive finite value"):
        normalize_spectra(
            [[1.0, 2.0]],
            phase_features=[[1]],
            phase_mask=[[0, 0]],
        )


def test_normalize_spectra_validates_shapes_and_numeric_values():
    with pytest.raises(ValueError, match="rectangular"):
        normalize_spectra([[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="numeric"):
        normalize_spectra([[1.0, "not-a-number"]])
    with pytest.raises(ValueError, match="finite"):
        normalize_spectra([[1.0, np.inf]])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_spectra([[2.0, -1.0]])
    with pytest.raises(ValueError, match="phase features"):
        normalize_spectra([[1.0, 2.0]], phase_mask=[[1, 1]])
    with pytest.raises(ValueError, match="Phase mask values must be binary"):
        normalize_spectra(
            [[1.0, 2.0]], phase_features=[[1]], phase_mask=[[1, 0.5]],
        )
    with pytest.raises(ValueError, match="exactly one active binary phase"):
        normalize_spectra(
            [[1.0, 2.0]], phase_features=[[1, 1]], phase_mask=[[1, 1], [1, 1]],
        )


def test_normalize_spectra_returns_finite_unit_sum_rows():
    normalized = np.asarray(
        normalize_spectra(
            [[0.0, 1.0, None], [2.0, 2.0, 4.0]],
            threshold=1e-8,
        ),
        dtype=object,
    )

    assert normalized[0, 2] is None
    assert sum(value for value in normalized[0] if value is not None) == pytest.approx(1)
    assert sum(normalized[1]) == pytest.approx(1)


def test_normalize_spectra_accepts_numpy_arrays():
    spectra = np.asarray([[1.0, 3.0], [2.0, 2.0]])

    normalized = normalize_spectra(spectra)

    np.testing.assert_allclose(normalized, [[0.25, 0.75], [0.5, 0.5]])
    assert normalize_spectra(np.empty((0, 2))) == []


def test_roundrobin_sid_does_not_mutate_ensemble_predictions():
    spectra = np.asarray(
        [[[0.0, 0.5], [1.0, 0.5], [np.nan, np.nan]]],
        dtype=float,
    )
    original = spectra.copy()

    result = roundrobin_sid(spectra, threshold=1e-8)

    np.testing.assert_equal(spectra, original)
    assert len(result) == 1
    assert np.isfinite(result[0])


def test_roundrobin_sid_validates_ensemble_and_missing_layout():
    with pytest.raises(ValueError, match="at least two ensemble"):
        roundrobin_sid(np.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="missing for all ensemble"):
        roundrobin_sid(np.asarray([[[1.0, np.nan], [0.5, 0.5]]]))
    with pytest.raises(ValueError, match="finite and positive"):
        roundrobin_sid(np.ones((1, 2, 2)), threshold=np.nan)
    with pytest.raises(ValueError, match="strictly positive"):
        roundrobin_sid(np.asarray([[[0.0, 0.5], [1.0, 0.5]]]))


@pytest.mark.parametrize(
    ('contents', 'message'),
    [
        ('', 'header row'),
        ('phase,bin\n', 'at least one phase row'),
        ('phase,bin\na,1,0\n', 'expected 2'),
        ('phase,bin,bin\na,1,0\n', 'uniquely named'),
        ('phase,bin\na,1\na,0\n', 'non-empty and unique'),
        ('phase,bin\na,2\n', 'only 0s and 1s'),
    ],
)
def test_load_phase_mask_rejects_ambiguous_or_malformed_csv(
    tmp_path, contents, message,
):
    path = tmp_path / 'phase_mask.csv'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        load_phase_mask(str(path))


def test_load_phase_mask_reads_rectangular_binary_values(tmp_path):
    path = tmp_path / 'phase_mask.csv'
    path.write_text('phase,a,b\nliquid,1,0\nsolid,0,1\n', encoding='utf-8')

    assert load_phase_mask(str(path)) == [[1, 0], [0, 1]]
