from collections import OrderedDict, namedtuple
import os
from typing import List, Optional, Tuple
from typing_extensions import Literal

import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.train import evaluate_predictions


FAKE_LOGGER = namedtuple('FakeLogger', ['info'])(info=lambda x: None)

DATASETS = OrderedDict()
DATASETS['qm7'] = {'metric': 'mae', 'type': 'regression'}
DATASETS['qm8'] = {'metric': 'mae', 'type': 'regression'}
DATASETS['qm9'] = {'metric': 'mae', 'type': 'regression'}
DATASETS['delaney'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['freesolv'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['lipo'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['pdbbind_full'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['pdbbind_core'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['pdbbind_refined'] = {'metric': 'rmse', 'type': 'regression'}
DATASETS['pcba'] = {'metric': 'prc-auc', 'type': 'classification'}
DATASETS['muv'] = {'metric': 'prc-auc', 'type': 'classification'}
DATASETS['hiv'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['bace'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['bbbp'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['tox21'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['toxcast'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['sider'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['clintox'] = {'metric': 'auc', 'type': 'classification'}
DATASETS['chembl'] = {'metric': 'auc', 'type': 'classification'}

# test if 1 is better than 2 (less error, higher auc)
COMPARISONS = [
    ('default', 'random_forest'),
    ('default', 'ffn_morgan'),
    ('default', 'ffn_morgan_count'),
    ('default', 'ffn_rdkit'),
    ('features_no_opt', 'default'),
    ('hyperopt_eval', 'default'),
    ('hyperopt_ensemble', 'default'),
    ('hyperopt_eval', 'features_no_opt'),
    ('hyperopt_ensemble', 'hyperopt_eval'),
    ('default', 'undirected'),
    ('default', 'atom_messages'),
    ('hyperopt_eval', 'compare_lsc_scaffold')
]

EXPERIMENTS = sorted({exp for comp in COMPARISONS for exp in comp})


class Args(Tap):
    preds_dir: str  # Path to a directory containing predictions
    split_type: Literal['random', 'scaffold']  # Split type


def load_preds_and_targets(preds_dir: str,
                           experiment: str,
                           dataset: str,
                           split_type: str) -> Tuple[Optional[np.ndarray],
                                                     Optional[np.ndarray]]:
    all_preds, all_targets = [], []
    num_folds = 0
    for fold in range(10):
        preds_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'preds.npy')
        targets_path = os.path.join(preds_dir, f'417_{experiment}', dataset, split_type, str(fold), 'targets.npy')

        if not (os.path.exists(preds_path) and os.path.exists(targets_path)):
            continue

        preds = np.load(preds_path, allow_pickle=False)
        targets = np.load(targets_path, allow_pickle=False)

        if preds.ndim < 2 or targets.ndim < 2:
            raise ValueError(
                f'Prediction and target arrays must each have at least two '
                f'dimensions ({preds_path}, {targets_path}).'
            )
        if not np.isfinite(preds).all():
            raise ValueError(f'Predictions contain NaN or infinity: {preds_path}')
        try:
            targets = np.asarray(targets, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Targets must be numeric: {targets_path}') from exc
        if np.isinf(targets).any():
            raise ValueError(f'Targets contain infinity: {targets_path}')

        all_preds.append(preds)
        all_targets.append(targets)

        num_folds += 1

    if num_folds not in [3, 10]:
        print(f'Did not find 3 or 10 preds/targets files for experiment "{experiment}" and dataset "{dataset}" and split type "{split_type}"')
        return None, None

    all_preds, all_targets = np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)

    if all_preds.shape != all_targets.shape:
        raise ValueError(
            f'Prediction shape {all_preds.shape} does not match target shape '
            f'{all_targets.shape} for {experiment}/{dataset}.'
        )

    return all_preds, all_targets


def compute_values(dataset: str,
                   preds: List[List[List[float]]],
                   targets: List[List[List[float]]]) -> List[float]:
    if dataset not in DATASETS:
        raise ValueError(f'Unknown benchmark dataset {dataset!r}.')
    if len(preds) == 0 or len(preds) != len(targets):
        raise ValueError(
            'Prediction and target groups must be non-empty and have equal length.'
        )
    if len(preds[0]) == 0 or len(preds[0][0]) == 0:
        raise ValueError('Prediction groups must contain at least one task.')
    num_tasks = len(preds[0][0])

    def normalize_missing_targets(target):
        numeric = np.asarray(target, dtype=float)
        normalized = numeric.astype(object)
        normalized[np.isnan(numeric)] = None
        return normalized

    values = [
        evaluate_predictions(
            preds=pred,
            targets=normalize_missing_targets(target),
            num_tasks=num_tasks,
            metrics=[DATASETS[dataset]['metric']],
            dataset_type=DATASETS[dataset]['type'],
            logger=FAKE_LOGGER
        )
        for pred, target in tqdm(zip(preds, targets), total=len(preds))
    ]

    metric = DATASETS[dataset]['metric']
    values = [np.nanmean(value[metric]) for value in values]

    return values


def wilcoxon_significance(preds_dir: str, split_type: str):
    print('dataset\t' + '\t'.join([f'{exp_1} vs {exp_2}' for exp_1, exp_2 in COMPARISONS]))

    for dataset in DATASETS:
        dataset_type = DATASETS[dataset]['type']

        # Compute values
        experiment_to_values = {}
        reference_targets = None
        for experiment in EXPERIMENTS:
            if experiment == 'compare_lsc_scaffold' and split_type != 'scaffold':
                continue

            preds, targets = load_preds_and_targets(preds_dir, experiment, dataset, split_type)  # num_molecules x num_targets

            if preds is None or targets is None:
                experiment_to_values[experiment] = None
                continue

            if reference_targets is None:
                reference_targets = np.array(targets, copy=True)
            elif not np.array_equal(targets, reference_targets, equal_nan=True):
                raise ValueError(
                    f'Target rows for {dataset}/{experiment} are not aligned '
                    'with the other experiments.'
                )

            if dataset_type == 'regression':
                preds, targets = [[pred] for pred in preds], [[target] for target in targets]
            else:
                preds, targets = np.array_split(preds, 30), np.array_split(targets, 30)

            values = compute_values(dataset, preds, targets)
            experiment_to_values[experiment] = values

        print(dataset, end='\t')

        # Compute p-values
        for experiment_1, experiment_2 in COMPARISONS:
            if 'compare_lsc_scaffold' in [experiment_1, experiment_2] and split_type != 'scaffold':
                continue

            values_1, values_2 = experiment_to_values[experiment_1], experiment_to_values[experiment_2]

            if values_1 is None or values_2 is None:
                print('Error', end='\t')
                continue

            if len(values_1) != len(values_2):
                raise ValueError(
                    f'Paired comparison lengths differ for {dataset}: '
                    f'{len(values_1)} != {len(values_2)}.'
                )

            # Remove non-finite metric groups while preserving paired order.
            finite_pairs = [
                (v_1, v_2)
                for v_1, v_2 in zip(values_1, values_2)
                if np.isfinite(v_1) and np.isfinite(v_2)
            ]
            if not finite_pairs:
                print('Error', end='\t')
                continue
            values_1, values_2 = zip(*finite_pairs)

            # test if error of 1 is less than error of 2
            if np.array_equal(values_1, values_2):
                pvalue = 1.0
            else:
                pvalue = wilcoxon(
                    values_1,
                    values_2,
                    alternative='less' if dataset_type == 'regression' else 'greater',
                ).pvalue
            print(pvalue, end='\t')
        print()


if __name__ == '__main__':
    args = Args().parse_args()

    wilcoxon_significance(
        preds_dir=args.preds_dir,
        split_type=args.split_type,
    )
