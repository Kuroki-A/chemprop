import os
from typing import List
from typing_extensions import Literal


from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np


ORDER = {
    name: index for index, name in enumerate([
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
        'chembl',
        'rppb',
        'sol',
        'rlm',
        'hpxr',
        'hpxr (class)',
        'benzene',
        'cyclohexane',
        'dichloromethane',
        'dmso',
        'ethanol',
        'ethyl acetate',
        'h2o',
        'octanol',
        'tetrahydrofuran',
        'toluene',
        'logp'
    ])
}


class Args(Tap):
    ckpts_dirs: List[str]  # Path to directories (one per dataset) with model save dirs
    split_type: Literal['random', 'scaffold']  # Split type, either "random" or "scaffold"


def aggregate_results(ckpts_dirs: List[str], split_type: str):
    print('Name\tMean\tStd\tNum files')

    ckpts_dirs = sorted(
        ckpts_dirs,
        key=lambda path: (
            ORDER.get(os.path.basename(os.path.normpath(path)), len(ORDER)),
            os.path.basename(os.path.normpath(path)),
        ),
    )

    for ckpts_dir in ckpts_dirs:
        name = os.path.basename(os.path.normpath(ckpts_dir))

        # Collect verbose.log files
        paths = []
        for root, _, files in os.walk(ckpts_dir):
            if split_type not in os.path.normpath(root).split(os.sep):
                continue
            paths += [os.path.join(root, fname) for fname in files if fname == 'verbose.log']
        paths.sort()

        # Process verbose.log files
        results = []
        invalid = False
        for path in paths:
            result = None
            with open(path) as rf:
                for line in rf:
                    if 'Overall test ' not in line or '=' not in line:
                        continue
                    try:
                        candidate = float(
                            line.split('=', 1)[1].split('+/-', 1)[0].strip()
                        )
                    except ValueError:
                        continue
                    if np.isfinite(candidate):
                        result = candidate
            if result is None:
                invalid = True
            else:
                results.append(result)

        if invalid or not results:
            mean, std = 'N/A', 'N/A'
        else:
            mean, std = np.mean(results), np.std(results)

        # Compute results
        print(f'{name}\t{mean}\t{std}\t{len(results)}')


if __name__ == '__main__':
    args = Args().parse_args()

    aggregate_results(
        ckpts_dirs=args.ckpts_dirs,
        split_type=args.split_type
    )
