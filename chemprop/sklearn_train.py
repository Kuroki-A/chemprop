from dataclasses import dataclass
from logging import Logger
import os
import pickle
import tempfile
from typing import Any, Dict, List, Union
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, SGDRegressor
from tqdm import trange, tqdm

from chemprop.args import SklearnTrainArgs
from chemprop.data import MoleculeDataset, get_data, split_data, get_task_names
from chemprop.features import get_features_generator
from chemprop.train import cross_validate, evaluate_predictions
from chemprop.train.run_training import validate_features_source_metadata
from chemprop.utils import save_smiles_splits


SKLEARN_BUNDLE_FORMAT = "chemprop-sklearn-bundle"
SKLEARN_BUNDLE_VERSION = 1


@dataclass
class SklearnModelBundle:
    """A versioned sklearn checkpoint, including one estimator per task when needed."""

    models: List[Any]
    train_args: Dict[str, Any]
    single_task: bool
    format: str = SKLEARN_BUNDLE_FORMAT
    version: int = SKLEARN_BUNDLE_VERSION


def _positive_class_probabilities(model, features) -> np.ndarray:
    """Returns positive-class probabilities, including one-class RF models."""
    try:
        probabilities = model.predict_proba(features)
    except AttributeError as error:
        raise ValueError(
            'This classification checkpoint cannot produce probabilities. '
            'Legacy SVM checkpoints trained without probability calibration must be retrained.'
        ) from error
    if isinstance(probabilities, list):
        columns = []
        for task_probabilities, classes in zip(probabilities, model.classes_):
            classes = np.asarray(classes)
            positive_indices = np.flatnonzero(classes == 1)
            if len(positive_indices) == 0:
                columns.append(np.zeros(len(task_probabilities), dtype=float))
            else:
                columns.append(task_probabilities[:, positive_indices[0]])
        return np.column_stack(columns)

    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) == 0:
        return np.zeros((len(probabilities), 1), dtype=float)
    return probabilities[:, positive_indices[0]].reshape(-1, 1)


def _predict_estimator(model,
                       model_type: str,
                       dataset_type: str,
                       features: List[np.ndarray]) -> List[List[float]]:
    """Predicts with one fitted sklearn estimator."""
    if len(features) == 0:
        return []

    if dataset_type == 'regression':
        preds = np.asarray(model.predict(features))

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
    elif dataset_type == 'classification':
        if model_type in {'random_forest', 'svm'}:
            preds = _positive_class_probabilities(model, features)
        else:
            raise ValueError(f'Model type "{model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{dataset_type}" not supported')

    return preds.tolist()


def predict(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC, SklearnModelBundle],
            model_type: str,
            dataset_type: str,
            features: List[np.ndarray]) -> List[List[float]]:
    """
    Predicts using a scikit-learn model.

    :param model: The trained scikit-learn model to make predictions with.
    :param model_type: The type of model.
    :param dataset_type: The type of dataset.
    :param features: The data features used as input for the model.
    :return: A list of lists of floats containing the predicted values.
    """
    if isinstance(model, SklearnModelBundle):
        if not model.models:
            raise ValueError('The sklearn checkpoint contains no fitted estimators.')
        if not model.single_task:
            return _predict_estimator(
                model.models[0], model_type, dataset_type, features
            )
        if len(features) == 0:
            return []

        task_columns = []
        for task_model in model.models:
            task_predictions = np.asarray(
                _predict_estimator(task_model, model_type, dataset_type, features),
                dtype=float,
            )
            if task_predictions.shape != (len(features), 1):
                raise ValueError(
                    'A single-task sklearn estimator returned an unexpected prediction shape.'
                )
            task_columns.append(task_predictions[:, 0])
        return np.column_stack(task_columns).tolist()

    return _predict_estimator(model, model_type, dataset_type, features)

def impute_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                   train_data: MoleculeDataset,
                   args: SklearnTrainArgs,
                   logger: Logger = None,
                   threshold: float = 0.5) -> List[float]:
    """
    Trains a single-task scikit-learn model, meaning a separate model is trained for each task.

    This is necessary if some tasks have None (unknown) values.

    :param model: The scikit-learn model to train.
    :param train_data: The training data.
    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 training the scikit-learn model.
    :param logger: A logger to record output.
    :param theshold: Threshold for classification tasks.
    :return: A list of list of target values.
    """
    num_tasks = train_data.num_tasks()
    new_targets=deepcopy(train_data.targets())
    
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
        
    debug('Imputation')
    
    for task_num in trange(num_tasks):
        impute_train_features = [features for features, targets in zip(train_data.features(), train_data.targets()) if targets[task_num] is None]
        if len(impute_train_features) > 0:
            observed = [(features, targets[task_num])
                        for features, targets in zip(train_data.features(), train_data.targets())
                        if targets[task_num] is not None]
            if not observed:
                task_name = args.task_names[task_num]
                raise ValueError(f'Sklearn task "{task_name}" has no training targets.')
            train_features, train_targets = zip(*observed)
            if args.impute_mode == 'single_task':
                imputation_model = deepcopy(model)
                imputation_model.fit(train_features, train_targets)
                impute_train_preds = predict(
                    model=imputation_model,
                    model_type=args.model_type,
                    dataset_type=args.dataset_type,
                    features=impute_train_features
                )
                impute_train_preds = [pred[0] for pred in impute_train_preds]
            elif args.impute_mode == 'median' and args.dataset_type == 'regression':
                impute_train_preds = [np.median(train_targets)] * len(impute_train_features)
            elif args.impute_mode == 'mean' and args.dataset_type == 'regression':
                impute_train_preds = [np.mean(train_targets)] * len(impute_train_features)
            elif args.impute_mode == 'frequent' and args.dataset_type == 'classification':
                integer_targets = np.asarray(train_targets, dtype=int)
                impute_train_preds = [
                    int(np.argmax(np.bincount(integer_targets)))
                ] * len(impute_train_features)
            elif args.impute_mode == 'linear' and args.dataset_type == 'regression':
                reg = SGDRegressor(alpha=0.01, random_state=args.seed).fit(train_features, train_targets)
                impute_train_preds = reg.predict(impute_train_features)
            elif args.impute_mode == 'linear' and args.dataset_type == 'classification':
                cls = SGDClassifier(random_state=args.seed).fit(train_features, train_targets)
                impute_train_preds = cls.predict(impute_train_features)
            else:
                raise ValueError("Invalid combination of imputation mode and dataset type.")   

            #Replace targets
            ctr = 0
            for i in range(len(new_targets)):
                if new_targets[i][task_num] is None:
                    value = impute_train_preds[ctr]
                    if args.dataset_type == 'classification':
                        value = int(value > threshold)
                    new_targets[i][task_num] = value
                    ctr += 1
                    
    return new_targets


def _fit_single_task_models(
    model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
    train_data: MoleculeDataset,
    args: SklearnTrainArgs,
) -> List[Any]:
    """Fits exactly one independent estimator for each task."""
    models = []
    for task_num in trange(train_data.num_tasks()):
        observed = [
            (features, targets[task_num])
            for features, targets in zip(train_data.features(), train_data.targets())
            if targets[task_num] is not None
        ]
        if not observed:
            task_name = args.task_names[task_num]
            raise ValueError(f'Sklearn task "{task_name}" has no training targets.')
        train_features, train_targets = zip(*observed)
        task_model = deepcopy(model)
        try:
            task_model.fit(train_features, train_targets)
        except ValueError as error:
            task_name = args.task_names[task_num]
            raise ValueError(
                f'Could not fit sklearn task "{task_name}": {error}'
            ) from error
        models.append(task_model)
    return models


def _evaluate_single_task_models(
    models: List[Any],
    data: MoleculeDataset,
    metrics: List[str],
    args: SklearnTrainArgs,
    logger: Logger = None,
) -> Dict[str, List[float]]:
    """Evaluates task-specific estimators without fitting them again."""
    scores = {metric: [] for metric in metrics}
    for task_num, task_model in enumerate(models):
        observed = [
            (features, targets[task_num])
            for features, targets in zip(data.features(), data.targets())
            if targets[task_num] is not None
        ]
        if not observed:
            for metric in metrics:
                scores[metric].append(float('nan'))
            continue

        data_features, data_targets = zip(*observed)
        data_preds = predict(
            model=task_model,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            features=data_features,
        )
        task_scores = evaluate_predictions(
            preds=data_preds,
            targets=[[target] for target in data_targets],
            num_tasks=1,
            metrics=metrics,
            dataset_type=args.dataset_type,
            logger=logger,
        )
        for metric in metrics:
            scores[metric].append(task_scores[metric][0])
    return scores


def _fit_multi_task_model(
    model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
    train_data: MoleculeDataset,
    args: SklearnTrainArgs,
    logger: Logger = None,
):
    """Fits one multi-output estimator once."""
    train_targets = train_data.targets()
    if args.impute_mode:
        train_targets = impute_sklearn(
            model=model, train_data=train_data, args=args, logger=logger
        )
    elif any(None in targets for targets in train_targets):
        raise ValueError(
            'Missing target values are not tolerated for multi-task sklearn models. '
            'Use --single_task or provide --impute_mode.'
        )

    if train_data.num_tasks() == 1:
        train_targets = [targets[0] for targets in train_targets]
    model.fit(train_data.features(), train_targets)
    return model


def _evaluate_multi_task_model(
    model,
    data: MoleculeDataset,
    metrics: List[str],
    args: SklearnTrainArgs,
    logger: Logger = None,
) -> Dict[str, List[float]]:
    """Evaluates a fitted multi-output estimator."""
    if len(data) == 0:
        return {
            metric: [float('nan')] * len(args.task_names) for metric in metrics
        }
    data_preds = predict(
        model=model,
        model_type=args.model_type,
        dataset_type=args.dataset_type,
        features=data.features(),
    )
    return evaluate_predictions(
        preds=data_preds,
        targets=data.targets(),
        num_tasks=len(args.task_names),
        metrics=metrics,
        dataset_type=args.dataset_type,
        logger=logger,
    )


def _build_sklearn_model(args: SklearnTrainArgs):
    """Builds an estimator whose outputs follow Chemprop's prediction contract."""
    if args.dataset_type == 'regression':
        if args.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=args.num_trees,
                n_jobs=-1,
                random_state=args.seed,
            )
        if args.model_type == 'svm':
            return SVR()
    elif args.dataset_type == 'classification':
        if args.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=args.num_trees,
                n_jobs=-1,
                class_weight=args.class_weight,
                random_state=args.seed,
            )
        if args.model_type == 'svm':
            return SVC(
                probability=True,
                class_weight=args.class_weight,
                random_state=args.seed,
            )

    raise ValueError(
        f'Model type "{args.model_type}" not supported for '
        f'dataset type "{args.dataset_type}"'
    )


def _save_sklearn_bundle(path: str, bundle: SklearnModelBundle) -> None:
    """Atomically saves a trusted sklearn bundle."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)), prefix='.sklearn-', suffix='.tmp'
    )
    try:
        with os.fdopen(descriptor, 'wb') as checkpoint_file:
            pickle.dump(bundle, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_sklearn_checkpoint(path: str) -> SklearnModelBundle:
    """Loads a new bundle or wraps a legacy single-estimator checkpoint."""
    with open(path, 'rb') as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)

    if isinstance(checkpoint, SklearnModelBundle):
        if checkpoint.format != SKLEARN_BUNDLE_FORMAT:
            raise ValueError(f'Unrecognized sklearn checkpoint format in "{path}".')
        if checkpoint.version != SKLEARN_BUNDLE_VERSION:
            raise ValueError(
                f'Unsupported sklearn checkpoint version {checkpoint.version!r} in "{path}".'
            )
        expected_models = (
            len(checkpoint.train_args.get('task_names') or [])
            if checkpoint.single_task else 1
        )
        if expected_models < 1 or len(checkpoint.models) != expected_models:
            raise ValueError(f'Invalid estimator count in sklearn checkpoint "{path}".')
        return checkpoint

    train_args = getattr(checkpoint, 'train_args', None)
    if not isinstance(train_args, dict):
        raise ValueError(f'Invalid or unsupported sklearn checkpoint "{path}".')
    return SklearnModelBundle(
        models=[checkpoint], train_args=train_args, single_task=False
    )


def single_task_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                        train_data: MoleculeDataset,
                        test_data: MoleculeDataset,
                        metrics: List[str],
                        args: SklearnTrainArgs,
                        logger: Logger = None) -> Dict[str, List[float]]:
    """Fits task-specific estimators once, saves them, and evaluates a split."""
    models = _fit_single_task_models(model, train_data, args)
    bundle = SklearnModelBundle(
        models=models, train_args=args.as_dict(), single_task=True
    )
    _save_sklearn_bundle(os.path.join(args.save_dir, 'model.pkl'), bundle)
    return _evaluate_single_task_models(models, test_data, metrics, args, logger)


def multi_task_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                       train_data: MoleculeDataset,
                       test_data: MoleculeDataset,
                       metrics: List[str],
                       args: SklearnTrainArgs,
                       logger: Logger = None) -> Dict[str, List[float]]:
    """Fits one multi-output estimator once, saves it, and evaluates a split."""
    model = _fit_multi_task_model(model, train_data, args, logger)
    bundle = SklearnModelBundle(
        models=[model], train_args=args.as_dict(), single_task=False
    )
    _save_sklearn_bundle(os.path.join(args.save_dir, 'model.pkl'), bundle)
    return _evaluate_multi_task_model(model, test_data, metrics, args, logger)


def run_sklearn(args: SklearnTrainArgs,
                data: MoleculeDataset,
                fold_num: int = None,
                logger: Logger = None):
    """
    Loads data, trains a scikit-learn model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 loading data and training the scikit-learn model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    legacy_call = fold_num is None or not isinstance(fold_num, int)
    if legacy_call and fold_num is not None and logger is None:
        logger = fold_num

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    skip_test_evaluation = bool(getattr(args, 'skip_test_evaluation', False))

    debug('Using preloaded data')
    args.task_names = get_task_names(path=args.data_path,
                                     smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns,
                                     ignore_columns=args.ignore_columns)

    if args.model_type == 'svm' and data.num_tasks() != 1:
        raise ValueError(f'SVM can only handle single-task data but found {data.num_tasks()} tasks')

    debug(f'Splitting data with seed {args.seed}')
    test_data = MoleculeDataset([])
    if args.separate_test_path and not skip_test_evaluation:
        test_data = get_data(
            path=args.separate_test_path,
            args=args,
            features_path=args.separate_test_features_path,
            atom_descriptors_path=args.separate_test_atom_descriptors_path,
            bond_descriptors_path=args.separate_test_bond_descriptors_path,
            phase_features_path=args.separate_test_phase_features_path,
            constraints_path=args.separate_test_constraints_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.task_names,
            loss_function=args.loss_function,
            logger=logger,
        )
        validate_features_source_metadata(
            data, test_data, 'separate test data'
        )
    if args.separate_val_path:
        val_data = get_data(
            path=args.separate_val_path,
            args=args,
            features_path=args.separate_val_features_path,
            atom_descriptors_path=args.separate_val_atom_descriptors_path,
            bond_descriptors_path=args.separate_val_bond_descriptors_path,
            phase_features_path=args.separate_val_phase_features_path,
            constraints_path=args.separate_val_constraints_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.task_names,
            loss_function=args.loss_function,
            logger=logger,
        )
        validate_features_source_metadata(
            data, val_data, 'separate validation data'
        )

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
    else:
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )

    if skip_test_evaluation:
        test_data = MoleculeDataset([])

    if args.save_smiles_splits and not skip_test_evaluation:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger
        )

    if skip_test_evaluation:
        debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | '
              f'val size = {len(val_data):,}')
    else:
        debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | '
              f'val size = {len(val_data):,} | test size = {len(test_data):,}')

    debug('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    fingerprint_datasets = [train_data, val_data]
    if not skip_test_evaluation:
        fingerprint_datasets.append(test_data)
    for dataset in fingerprint_datasets:
        for datapoint in tqdm(dataset, total=len(dataset)):
            for s in datapoint.smiles:
                datapoint.extend_features(morgan_fingerprint(mol=s, radius=args.radius, num_bits=args.num_bits))

    debug('Building model')
    model = _build_sklearn_model(args)

    debug(model)

    debug('Training')
    if args.single_task:
        fitted_models = _fit_single_task_models(model, train_data, args)
        bundle = SklearnModelBundle(
            models=fitted_models,
            train_args=args.as_dict(),
            single_task=True,
        )
        valid_scores = _evaluate_single_task_models(
            fitted_models, val_data, args.metrics, args, logger
        )
    else:
        fitted_model = _fit_multi_task_model(model, train_data, args, logger)
        bundle = SklearnModelBundle(
            models=[fitted_model],
            train_args=args.as_dict(),
            single_task=False,
        )
        valid_scores = _evaluate_multi_task_model(
            fitted_model, val_data, args.metrics, args, logger
        )

    # The exact estimator(s) used for both validation and test prediction are
    # persisted once. In particular, validation scoring never triggers a
    # second fit whose state could diverge from the saved model.
    _save_sklearn_bundle(os.path.join(args.save_dir, 'model.pkl'), bundle)

    if skip_test_evaluation:
        test_scores = {}
    elif args.single_task:
        test_scores = _evaluate_single_task_models(
            fitted_models, test_data, args.metrics, args, logger
        )
    else:
        test_scores = _evaluate_multi_task_model(
            fitted_model, test_data, args.metrics, args, logger
        )

    for metric in args.metrics:
        valid_metric_scores = np.asarray(valid_scores[metric], dtype=float)
        valid_mean = (
            float('nan')
            if np.all(np.isnan(valid_metric_scores))
            else np.nanmean(valid_metric_scores)
        )
        info(f'Validation {metric} = {valid_mean}')
        if not skip_test_evaluation:
            test_metric_scores = np.asarray(test_scores[metric], dtype=float)
            test_mean = (
                float('nan')
                if np.all(np.isnan(test_metric_scores))
                else np.nanmean(test_metric_scores)
            )
            info(f'Test {metric} = {test_mean}')

    if legacy_call:
        return test_scores

    return valid_scores, test_scores


def sklearn_train() -> None:
    """Parses scikit-learn training arguments and trains a scikit-learn model.

    This is the entry point for the command line command :code:`sklearn_train`.
    """
    cross_validate(args=SklearnTrainArgs().parse_args(), train_func=run_sklearn)
