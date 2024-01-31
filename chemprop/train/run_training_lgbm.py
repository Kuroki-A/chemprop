import json
from logging import Logger
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .loss_functions import get_loss_func
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel, MoleculeModelEncoder
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean

import pickle
from sklearn.metrics import roc_curve, auc, mean_squared_error
import lightgbm as lgb

def run_training_lgbm(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_descriptors_path=args.separate_test_bond_descriptors_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             constraints_path=args.separate_test_constraints_path,
                             smiles_columns=args.smiles_columns,
                             loss_function=args.loss_function,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_descriptors_path=args.separate_val_bond_descriptors_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            constraints_path=args.separate_val_constraints_path,
                            smiles_columns=args.smiles_columns,
                            loss_function=args.loss_function,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=args.split_sizes,
                                              key_molecule_index=args.split_key_molecule,
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=args.split_sizes,
                                             key_molecule_index=args.split_key_molecule,
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     key_molecule_index=args.split_key_molecule,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        train_class_sizes = get_class_sizes(train_data, proportion=False)
        args.train_class_sizes = train_class_sizes

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            constraints_path=args.constraints_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_descriptor_scaling and args.bond_descriptors is not None:
        bond_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
        val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if len(val_data) == 0:
        raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
            a validation set is required to conduct early stopping according to the selected evaluation metric. This \
            may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

    if len(test_data) == 0:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
        empty_test_set = True
    else:
        empty_test_set = False


    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        if args.is_atom_bond_targets:
            scaler = None
            atom_bond_scaler = train_data.normalize_atom_bond_targets()
        else:
            scaler = train_data.normalize_targets()
            atom_bond_scaler = None
        args.spectra_phase_mask = None
    elif args.dataset_type == 'spectra':
        debug('Normalizing spectra and excluding spectra regions based on phase')
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
        atom_bond_scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None
        atom_bond_scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    elif args.is_atom_bond_targets:
        sum_test_preds = []
        for tb in zip(*test_data.targets()):
            tb = np.concatenate(tb)
            sum_test_preds.append(np.zeros((tb.shape[0], 1)))
        sum_test_preds = np.array(sum_test_preds, dtype=object)
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers
        
    encoder = MoleculeModelEncoder(args)
    encoder = encoder.to(args.device)

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=len(train_data),
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=len(val_data),
        num_workers=num_workers
    )
    
    if empty_test_set:
        info(f'Model provided with no test set, no metric evaluation will be performed.')
    else:
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=len(test_data),
            num_workers=num_workers
        )
        
        for batch in tqdm(test_data_loader, total=len(test_data_loader), leave=False):
            test_mol_batch, test_features_batch, test_target_batch = batch.batch_graph(), batch.features(), batch.targets()
    
        test_features = encoder(test_mol_batch, test_features_batch)
        test_features = test_features.to('cpu').detach().numpy().copy()
        test_target_batch = [x for row in test_target_batch for x in row]

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    best_valid_scores = defaultdict(list)
    test_preds = []
    
    if args.metric == 'auc':
        params = {"objective": "binary",
                  "metric": "auc",
                  "seed": 46}
    elif args.metric == 'rmse':
        params = {"objective": "regression",
                  "metric": "rmse",
                  "seed": 46}

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        
        for batch in tqdm(train_data_loader, total=len(train_data_loader), leave=False):
            train_mol_batch, train_features_batch, train_target_batch = batch.batch_graph(), batch.features(), batch.targets()
            
        for batch in tqdm(val_data_loader, total=len(val_data_loader), leave=False):
            val_mol_batch, val_features_batch, val_target_batch = batch.batch_graph(), batch.features(), batch.targets()      
            
        train_features = encoder(train_mol_batch, train_features_batch)
        val_features = encoder(val_mol_batch, val_features_batch)
        
        train_features = train_features.to('cpu').detach().numpy().copy()
        val_features = val_features.to('cpu').detach().numpy().copy()
        
        train_target_batch = [x for row in train_target_batch for x in row]
        val_target_batch = [x for row in val_target_batch for x in row]
        
        lgb_train = lgb.Dataset(train_features, train_target_batch, weight=None)
        lgb_eval = lgb.Dataset(val_features, val_target_batch, weight=None)

        model = lgb.train(params=params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      callbacks=[
                        lgb.early_stopping(stopping_rounds=10, verbose=True)
                      ])

        val_pred = model.predict(val_features).reshape(-1, 1)
        
        if not empty_test_set:
            test_pred = model.predict(test_features).reshape(-1, 1)
            test_preds.append(test_pred)
        
        if args.metric == 'auc':
            fpr, tpr, thresholds = roc_curve(val_target_batch, val_pred, pos_label=1)
            best_score = auc(fpr, tpr)
            best_valid_scores[args.metric].append(best_score)
            
        elif args.metric == 'rmse':
            best_score = mean_squared_error(val_target_batch, val_pred, squared=False)
            best_valid_scores[args.metric].append(best_score)

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f}')
        
        model_file = f'./{args.save_dir}/lgbm_{model_idx}.pkl'
        pickle.dump(model, open(model_file, 'wb'))
        
        args.save(f'./{args.save_dir}/args_{model_idx}.json')

    if empty_test_set:
        ensemble_test_scores = {
            metric: [np.nan for task in args.task_names] for metric in args.metrics
        }
    else:
        test_preds_mean = sum(test_preds) / len(test_preds)
    
        ensemble_test_scores = {}
        if args.metric == 'auc':
            fpr, tpr, thresholds = roc_curve(test_target_batch, test_preds_mean, pos_label=1)
            ensemble_test_scores[args.metrics[0]] = [auc(fpr, tpr)]

        elif args.metric == 'rmse':
            ensemble_test_scores[args.metrics] = [mean_squared_error(test_target_batch, test_preds_mean, squared=False)]

    return dict(best_valid_scores), ensemble_test_scores
