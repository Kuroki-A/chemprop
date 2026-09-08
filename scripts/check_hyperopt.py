import os
from typing import List
from typing_extensions import Literal


from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    ckpts_dirs: List[str]  # Paths to directory containing hyperopt config.json files in directories labelled by fold number (0, 1, ...)
    split_type: Literal['random', 'scaffold']  # Split type, either "random" or "scaffold"
    num_folds: int = 10  # Number of folds


def main(ckpts_dirs: List[str], split_type: str, num_folds: int):
    if num_folds <= 0:
        raise ValueError('num_folds must be positive.')
    for ckpts_dir in ckpts_dirs:
        # Find all config.json files
        fnames = []
        for root, _, files in os.walk(ckpts_dir):
            if split_type not in os.path.normpath(root).split(os.sep):
                continue
            fnames += [os.path.join(root, fname) for fname in files if fname == 'config.json']

        # Print out complete and incomplete
        complete = set()
        for fname in fnames:
            directory = os.path.basename(os.path.dirname(fname))
            try:
                fold = int(directory)
            except ValueError:
                print(f'Ignoring config outside a numeric fold directory: {fname}')
                continue
            if 0 <= fold < num_folds:
                complete.add(fold)
        incomplete = set(range(num_folds)) - complete

        print(os.path.basename(ckpts_dir))
        print(f'complete = {" ".join(str(fold) for fold in sorted(complete))}')
        print(f'incomplete = {" ".join(str(fold) for fold in sorted(incomplete))}')
        print()


if __name__ == '__main__':
    args = Args().parse_args()

    main(
        ckpts_dirs=args.ckpts_dirs,
        split_type=args.split_type,
        num_folds=args.num_folds
    )
