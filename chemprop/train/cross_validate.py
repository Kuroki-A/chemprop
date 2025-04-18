from collections import defaultdict
import csv
import json
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple
import subprocess

import numpy as np
import pandas as pd

import torch

from .run_training import run_training
from .run_training_lgbm import run_training_lgbm

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters

from distutils.version import LooseVersion

@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs,
                   train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_columns=args.smiles_columns,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns,
        loss_function=args.loss_function,
    )

    args.quantiles = [args.quantile_loss_alpha / 2] * (args.num_tasks // 2) + [1 - args.quantile_loss_alpha / 2] * (
        args.num_tasks // 2
    )

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    try:
        args.save(os.path.join(args.save_dir, 'args.json'))
    except subprocess.CalledProcessError:
        debug('Could not write the reproducibility section of the arguments to file, thus omitting this section.')
        args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    # set explicit H option and reaction option
    reset_featurization_parameters(logger=logger)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if args.reaction:
        set_reaction(args.reaction, args.reaction_mode)
    elif args.reaction_solvent:
        set_reaction(True, args.reaction_mode)

    # Get data
    if args.use_cache:
        features_names = 'MoleculeDataset'
        if args.features_generator is not None:
            for name in args.features_generator:
                features_names += '_' + str(name)

        is_file = os.path.isfile(f'{features_names}.pt')
        if not is_file:
            debug('Loading data')
            data = get_data(
                path=args.data_path,
                args=args,
                logger=logger,
                skip_none_targets=True,
                data_weights_path=args.data_weights_path
            )

            torch.save(data, f'{features_names}.pt')
        else:
            debug('Loading previously created data')
            if LooseVersion(torch.__version__) >= LooseVersion("2.6"):
                data = torch.load(f'{features_names}.pt', weights_only=False)
            else:
                data = torch.load(f'{features_names}.pt')
    else:
        debug('Loading data')
        data = get_data(
            path=args.data_path,
            args=args,
            logger=logger,
            skip_none_targets=True,
            data_weights_path=args.data_weights_path
        )
    
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_descriptors == 'descriptor':
        args.bond_descriptors_size = data.bond_descriptors_size()
    elif args.bond_descriptors == 'feature':
        args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError('The number of provided target weights must match the number and order of the prediction tasks')

    # Run training on different random seeds for each fold
    all_valid_scores = defaultdict(list)
    all_test_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()

        # If resuming experiment, load results from trained models
        test_scores_path = os.path.join(args.save_dir, 'test_scores.json')
        if args.resume_experiment and os.path.exists(test_scores_path):
            print('Loading scores')
            with open(test_scores_path) as f:
                model_test_scores = json.load(f)
        # Otherwise, train the models
        else:
            model_valid_scores, model_test_scores = train_func(args, data, fold_num, logger)

        for metric, scores in model_valid_scores.items():
            all_valid_scores[metric].append(scores)
        for metric, scores in model_test_scores.items():
            all_test_scores[metric].append(scores)
    
    all_valid_scores = dict(all_valid_scores)
    all_test_scores = dict(all_test_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_valid_scores.items():
        all_valid_scores[metric] = np.array(scores)
    for metric, scores in all_test_scores.items():
        all_test_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    contains_nan_scores = False
    for fold_num in range(args.num_folds):
        for metric, scores in all_test_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = '
                 f'{multitask_mean(scores=scores[fold_num], metric=metric, ignore_nan_metrics=args.ignore_nan_metrics):.6f}')

            if args.show_individual_scores:
                if args.loss_function == "quantile_interval" and metric == "quantile":
                    num_tasks = len(args.task_names) // 2
                    task_names = args.task_names[:num_tasks]
                    task_names = [f"{task_name} lower" for task_name in task_names] + [
                                  f"{task_name} upper" for task_name in task_names]
                else:
                    task_names = args.task_names

                for task_name, score in zip(task_names, scores[fold_num]):
                    info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')
                    if np.isnan(score):
                        contains_nan_scores = True

    # Report scores across folds
    for metric, scores in all_valid_scores.items():
        avg_scores = multitask_mean(
            scores=scores,
            axis=1,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics
        )  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall valid {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall valid {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    for metric, scores in all_test_scores.items():
        avg_scores = multitask_mean(
            scores=scores,
            axis=1,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics
        )  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    if contains_nan_scores:
        info("The metric scores observed for some fold test splits contain 'nan' values. \
            This can occur when the test set does not meet the requirements \
            for a particular metric, such as having no valid instances of one \
            task in the test set or not having positive examples for some classification metrics. \
            Before v1.5.1, the default behavior was to ignore nan values in individual folds or tasks \
            and still return an overall average for the remaining folds or tasks. The behavior now \
            is to include them in the average, converting overall average metrics to 'nan' as well.")

    # Save scores 
    '''
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        if args.dataset_type == 'spectra': # spectra data type has only one score to report
            row = ['spectra']
            for metric, scores in all_test_scores.items():
                task_scores = scores[:,0]
                mean, std = np.mean(task_scores), np.std(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
        else: # all other data types, separate scores by task
            if args.loss_function == "quantile_interval" and metric == "quantile":
                num_tasks = len(args.task_names) // 2
                task_names = args.task_names[:num_tasks]
                task_names = [f"{task_name} (lower quantile)" for task_name in task_names] + [
                                f"{task_name} (upper quantile)" for task_name in task_names]
            else:
                task_names = args.task_names

            for task_num, task_name in enumerate(task_names):
                row = [task_name]
                for metric, scores in all_test_scores.items():
                    task_scores = scores[:, task_num]
                    mean, std = np.mean(task_scores), np.std(task_scores)
                    row += [mean, std] + task_scores.tolist()
                writer.writerow(row)
    '''
    
    # Determine mean and std score of main metric
    if args.data_type == 'validation':
        avg_scores = multitask_mean(
        scores=all_valid_scores[args.metric],
        metric=args.metric, axis=1,
        ignore_nan_metrics=args.ignore_nan_metrics
        )
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
    elif args.data_type == 'test':
        avg_scores = multitask_mean(
        scores=all_test_scores[args.metric],
        metric=args.metric, axis=1,
        ignore_nan_metrics=args.ignore_nan_metrics
        )
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
    else:
        raise ValueError(f'"{args.optimize}" is not supported for hyperparameter optimization.')

    # Optionally merge and save test preds
    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                                  for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    args=TrainArgs().parse_args()
    
    if args.model_type == 'FFN':
        cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
    elif args.model_type == 'lgbm':
        cross_validate(args=TrainArgs().parse_args(), train_func=run_training_lgbm)
