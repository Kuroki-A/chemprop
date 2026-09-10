from typing import List
import csv

from tqdm import trange
import numpy as np


def normalize_spectra(spectra: List[List[float]], phase_features: List[List[float]] = None, phase_mask: List[List[float]] = None, batch_size: int = 50, excluded_sub_value: float = None, threshold: float = None) -> List[List[float]]:
    """
    Function takes in spectra and normalize them to sum values to 1. If provided with phase mask information, will remove excluded spectrum regions.

    :param spectra: Input spectra with shape (num_spectra, spectrum_length).
    :param phase_features: The collection phase of spectrum with shape (num_spectra, num_phases).
    :param phase_mask: A mask array showing where in each phase feature to include in predictions and training with shape (num_phases, spectrum_length)
    :param batch_size: The size of batches to carry out the normalization operation in.
    :param exlcuded_sub_value: Excluded values are replaced with this object, usually None or nan.
    :param threshold: Spectra values below threshold are replaced with threshold to remove negative or zero values.
    :return: List form array of spectra with shape (num_spectra, spectrum length) with exlcuded values converted to nan.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Spectra normalization batch_size must be a positive integer.')
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('Spectra normalization threshold must be finite and positive.')
    # NumPy arrays do not define a scalar truth value when they contain more
    # than one element, so do not use ``if not spectra`` here. Prediction
    # evaluation passes a NumPy array to this public helper.
    if spectra is None or len(spectra) == 0:
        return []

    spectrum_width = len(spectra[0])
    if spectrum_width == 0 or any(len(spectrum) != spectrum_width for spectrum in spectra):
        raise ValueError('Spectra must form a non-empty rectangular matrix.')

    normalized_spectra = []
    phase_exclusion = phase_mask is not None and phase_features is not None
    if phase_mask is not None and phase_features is None:
        raise ValueError('A phase mask requires phase features for every spectrum.')
    if phase_exclusion:
        phase_mask = np.asarray(phase_mask)
        phase_features = np.asarray(phase_features)
        if phase_mask.ndim != 2 or phase_mask.shape[1] != spectrum_width:
            raise ValueError('Phase mask width must match the spectrum width.')
        if phase_features.shape != (len(spectra), phase_mask.shape[0]):
            raise ValueError(
                'Phase features must have one row per spectrum and one column per phase.'
            )
        if not np.all(np.isfinite(phase_mask)) or not np.all(np.isfinite(phase_features)):
            raise ValueError('Phase masks and phase features must contain only finite values.')
        if np.any((phase_mask != 0) & (phase_mask != 1)):
            raise ValueError('Phase mask values must be binary (0 or 1).')
        if np.any((phase_features != 0) & (phase_features != 1)) or np.any(
            np.sum(phase_features, axis=1) != 1
        ):
            raise ValueError(
                'Phase features must contain exactly one active binary phase '
                'per spectrum.'
            )
    
    num_iters, iter_step = len(spectra), batch_size

    for i in trange(0, num_iters, iter_step):
        # prepare batch
        batch_spectra = spectra[i:i + iter_step]
        batch_mask = np.asarray([[x is not None for x in b] for b in batch_spectra])
        try:
            batch_spectra = np.asarray(
                [[0 if x is None else x for x in b] for b in batch_spectra],
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError('Spectra must contain numeric or missing values.') from error
        if not np.all(np.isfinite(batch_spectra[batch_mask])):
            raise ValueError('Observed spectrum values must be finite.')
        if threshold is None and np.any(batch_spectra[batch_mask] < 0):
            raise ValueError(
                'Observed spectrum values must be non-negative when no threshold is used.'
            )
        if phase_exclusion:
            batch_phases = phase_features[i:i + iter_step]

        # exclude mask and apply threshold
        if threshold is not None:
            batch_spectra[batch_spectra < threshold] = threshold
        if phase_exclusion:
            batch_phase_mask = np.matmul(batch_phases, phase_mask).astype('bool')
            batch_mask = np.logical_and(batch_mask, batch_phase_mask)
        batch_spectra[~batch_mask] = 0
        
        # normalize to sum to 1
        sum_spectra = np.sum(batch_spectra, axis=1, keepdims=True)
        empty_rows = np.flatnonzero(sum_spectra[:, 0] <= 0)
        if empty_rows.size:
            absolute_rows = (empty_rows + i).tolist()
            raise ValueError(
                'Each spectrum must retain at least one positive finite value; '
                f'invalid row indices: {absolute_rows}.'
            )
        batch_spectra = batch_spectra / sum_spectra

        # Collect vectors and revert excluded values to None
        batch_spectra = batch_spectra.astype('object')
        batch_spectra[~batch_mask] = excluded_sub_value
        batch_spectra = batch_spectra.tolist()
        normalized_spectra.extend(batch_spectra)
    
    return normalized_spectra


def roundrobin_sid(spectra: np.ndarray, threshold: float = None) -> List[float]:
    """
    Takes a block of input spectra and makes a pairwise comparison between each of the input spectra for a given molecule,
    returning a list of the spectral informations divergences. To be used evaluating the variation between an ensemble of model spectrum predictions.

    :spectra: A 3D array containing each of the spectra to be compared. Shape of (num_spectra, spectrum_length, ensemble_size)
    :threshold: SID calculation requires positive values in each position, this value is used to replace any zero or negative values.
    :return: A list of average pairwise SID len (num_spectra)
    """
    try:
        spectra_array = np.asarray(spectra, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Round-robin spectra must contain numeric values.') from error
    if spectra_array.ndim != 3 or spectra_array.shape[1] == 0:
        raise ValueError(
            'Round-robin spectra must have shape '
            '(num_spectra, spectrum_length, ensemble_size).'
        )

    ensemble_size = spectra_array.shape[2]
    if ensemble_size < 2:
        raise ValueError('Round-robin SID requires at least two ensemble members.')
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('Round-robin SID threshold must be finite and positive.')

    ensemble_heads, ensemble_tails = np.triu_indices(ensemble_size, k=1)
    ensemble_sids = []
    for spectrum_index in range(len(spectra_array)):
        # Work on a copy: the caller commonly retains individual ensemble
        # predictions for output, and thresholding must not change those values.
        spectrum = spectra_array[spectrum_index].copy()
        nan_values = np.isnan(spectrum)
        excluded_bins = np.all(nan_values, axis=1)
        if np.any(nan_values & ~excluded_bins[:, None]):
            raise ValueError(
                'Every excluded spectral bin must be missing for all ensemble members.'
            )
        observed = spectrum[~excluded_bins]
        if observed.size == 0:
            raise ValueError('Each spectrum must contain at least one observed bin.')
        if not np.all(np.isfinite(observed)):
            raise ValueError('Observed round-robin spectra values must be finite.')
        if threshold is not None:
            spectrum[~excluded_bins] = np.maximum(observed, threshold)
        elif np.any(observed <= 0):
            raise ValueError(
                'Round-robin SID requires strictly positive values when no '
                'threshold is provided.'
            )

        spectrum[excluded_bins] = 1
        heads = spectrum[:, ensemble_heads]
        tails = spectrum[:, ensemble_tails]
        pairwise_loss = (
            heads * np.log(heads / tails)
            + tails * np.log(tails / heads)
        )
        pairwise_loss[excluded_bins] = 0
        sid = float(np.mean(np.sum(pairwise_loss, axis=0)))
        if not np.isfinite(sid):
            raise ValueError('Round-robin SID produced a non-finite value.')
        ensemble_sids.append(sid)
    return ensemble_sids


def load_phase_mask(path: str) -> List[List[int]]:
    """
    Loads in a matrix used to mark sections of spectra as untrainable due to interference caused by particular phases.
    Ignore those spectra regions in training and prediciton.

    :param path: Path to a csv file containing the phase mask in shape (num_phases, spectrum_length) with 1s indicating inclusion and 0s indicating exclusion.
    :return: A list form array of the phase mask.
    """
    if path is None:
        return None

    data = []
    phase_names = set()
    with open(path, 'r', newline='', encoding='utf-8-sig') as rf:
        reader = csv.reader(rf)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError('Phase mask file must contain a header row.') from error
        if len(header) < 2 or len(header) != len(set(header)):
            raise ValueError(
                'Phase mask header must contain a phase-name column and at '
                'least one uniquely named spectrum column.'
            )
        for row_number, line in enumerate(reader, start=2):
            if len(line) != len(header):
                raise ValueError(
                    f'Phase mask CSV row {row_number} has {len(line)} columns; '
                    f'expected {len(header)}.'
                )
            phase_name = line[0]
            if phase_name == '' or phase_name in phase_names:
                raise ValueError(
                    'Phase mask phase names must be non-empty and unique.'
                )
            phase_names.add(phase_name)
            if any(value not in {'0', '1'} for value in line[1:]):
                raise ValueError(
                    'Phase mask must contain only 0s and 1s, with 0s '
                    'indicating exclusion regions.'
                )
            data.append([int(value) for value in line[1:]])
    if not data:
        raise ValueError('Phase mask file must contain at least one phase row.')
    return data
