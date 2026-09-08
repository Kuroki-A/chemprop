import os
import shutil

import h5py
import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


DATASETS = [
    'qm7',
    'qm8',
    'qm9',
    'delaney',
    'freesolv',
    'lipo',
    'pdbbind_full',
    'pdbbind_core',
    'pdbbind_refined',
    'pcba',
    'muv',
    'hiv',
    'bace',
    'bbbp',
    'tox21',
    'toxcast',
    'sider',
    'clintox',
    'chembl'
]


class Args(Tap):
    lsc_dir: str  # Path to directory in lsc save format
    ckpt_dir: str  # Path to directory with targets saved in our format
    save_dir: str  # Path to directory where lsc files will be saved in our format


def lsc_to_our_format(lsc_dir: str, ckpt_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for dataset in DATASETS:
        print(dataset, end='\t')

        success = 0

        # Convert preds and copy over preds and targets
        for fold in range(10):
            lsc_preds_path = os.path.join(lsc_dir, dataset, 'test', f'fold_{fold}', 'semi', 'o0003.evalPredict.hdf5')
            ckpt_targets_path = os.path.join(ckpt_dir, dataset, 'scaffold', str(fold), 'targets.npy')

            if not (os.path.exists(lsc_preds_path) and os.path.exists(ckpt_targets_path)):
                continue

            save_fold_dir = os.path.join(save_dir, dataset, 'scaffold', str(fold))
            os.makedirs(save_fold_dir, exist_ok=True)

            save_preds_path = os.path.join(save_fold_dir, 'preds.npy')
            save_targets_path = os.path.join(save_fold_dir, 'targets.npy')

            # Validate alignment before publishing either output file.
            targets = np.load(ckpt_targets_path, allow_pickle=False)
            with h5py.File(lsc_preds_path, 'r') as preds_file:
                if 'predictions' not in preds_file:
                    raise ValueError(
                        f'HDF5 file does not contain a predictions dataset: '
                        f'{lsc_preds_path}'
                    )
                preds = np.asarray(preds_file['predictions'])
            if preds.shape != targets.shape:
                raise ValueError(
                    f'Prediction shape {preds.shape} does not match target shape '
                    f'{targets.shape} for {dataset} fold {fold}.'
                )
            if not np.issubdtype(preds.dtype, np.number) or not np.isfinite(preds).all():
                raise ValueError(
                    f'Predictions must be numeric and finite: {lsc_preds_path}'
                )

            shutil.copy2(ckpt_targets_path, save_targets_path)
            np.save(save_preds_path, preds)

            success += 1

        print(success)


if __name__ == '__main__':
    args = Args().parse_args()

    lsc_to_our_format(
        lsc_dir=args.lsc_dir,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir
    )
