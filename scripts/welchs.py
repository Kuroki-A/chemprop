from typing import List
from typing_extensions import Literal

import numpy as np
from scipy import stats
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


class Args(Tap):
    mean1: List[float]  # Means of distributions of 1st model
    mean2: List[float]  # Means of distributions of 2nd model
    std1: List[float]  # Standard deviations of distributions of 1st model
    std2: List[float]  # Standard deviations of distributions of 2nd model
    nobs1: List[int]  # Number of observations each mean/std is constructed from in 1st model
    nobs2: List[int]  # Number of observations each mean/std is constructed from in 2nd model
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'  # Alternative hypothesis for model 1 versus model 2


def welchs(mean1: List[float],  # mean performance across folds for each dataset (model 1)
           std1: List[float],  # standard deviation performance across folds for each dataset (model 1)
           nobs1: List[int],  # number of CV folds for each dataset (model 1)
           mean2: List[float],  # mean performance across folds for each dataset (model 2)
           std2: List[float],  # standard deviation performance across folds for each dataset (model 2)
           nobs2: List[int],  # number of CV folds for each dataset (model 2)
           alternative: str = 'two-sided'):
    # Expand one number of observations to all
    if len(nobs1) == 1:
        nobs1 = nobs1 * len(mean1)

    if len(nobs2) == 1:
        nobs2 = nobs2 * len(mean2)

    lengths = [len(mean1), len(std1), len(nobs1), len(mean2), len(std2), len(nobs2)]
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError(
            f'All input lists must have the same positive length; got {lengths}.'
        )
    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError('alternative must be two-sided, less, or greater.')
    numeric = np.asarray([mean1, std1, mean2, std2], dtype=float)
    if not np.isfinite(numeric).all() or np.any(np.asarray([std1, std2]) < 0):
        raise ValueError('Means/stds must be finite and standard deviations non-negative.')
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 1 for n in nobs1 + nobs2):
        raise ValueError('Every observation count must be an integer greater than 1.')

    # Convert from population standard deviation to sample standard deviation
    std1 = [s * np.sqrt(n / (n - 1)) for s, n in zip(std1, nobs1)]
    std2 = [s * np.sqrt(n / (n - 1)) for s, n in zip(std2, nobs2)]

    # Compute Welch's t-test p-values for each dataset based on mean, standard deviation, and number of observations
    pvalues = []
    for m1, s1, n1, m2, s2, n2 in zip(
        mean1, std1, nobs1, mean2, std2, nobs2
    ):
        if s1 == 0 and s2 == 0:
            if m1 == m2:
                pvalue = 1.0
            elif alternative == 'two-sided':
                pvalue = 0.0
            elif alternative == 'less':
                pvalue = 0.0 if m1 < m2 else 1.0
            else:
                pvalue = 0.0 if m1 > m2 else 1.0
        else:
            pvalue = stats.ttest_ind_from_stats(
                mean1=m1,
                std1=s1,
                nobs1=n1,
                mean2=m2,
                std2=s2,
                nobs2=n2,
                equal_var=False,
                alternative=alternative,
            ).pvalue
        pvalues.append(float(pvalue))

    # Print Welch's p-values
    print('\n'.join(f'{pvalue:.4e}' for pvalue in pvalues))

    # Chi-squared statistic
    if any(pvalue == 0 for pvalue in pvalues):
        chisquare, pvalue = float('inf'), 0.0
    else:
        chisquare, pvalue = stats.combine_pvalues(pvalues, method='fisher')
    print(f'X^2  = {chisquare}')

    # Degrees of freedom
    df = 2 * len(pvalues)
    print(f'df = {df}')

    # Print p-value
    print(f'p = {pvalue:.4e}')


if __name__ == '__main__':
    args = Args().parse_args()

    welchs(
        mean1=args.mean1,
        std1=args.std1,
        nobs1=args.nobs1,
        mean2=args.mean2,
        std2=args.std2,
        nobs2=args.nobs2,
        alternative=args.alternative,
    )
