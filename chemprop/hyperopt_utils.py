from chemprop.args import HyperoptArgs
from copy import deepcopy
import fcntl
import os
import pickle
import tempfile
from typing import List, Dict, Tuple
import json
import logging

from hyperopt import Trials, hp
import numpy as np

from chemprop.constants import HYPEROPT_SEED_FILE_NAME
from chemprop.utils import makedirs, multitask_mean


def _load_manual_validation_scores(trial_dir: str, trial_args: Dict) -> Tuple[float, float]:
    """Loads and aggregates validation scores for a manual hyperopt trial.

    Test scores are intentionally never used as a fallback: doing so would
    leak held-out labels into hyperparameter selection. Older manual runs must
    be rerun (or supplied with their per-fold ``valid_scores.json`` files).
    """
    metric = trial_args["metric"]
    fold_scores = []
    for fold_num in range(trial_args["num_folds"]):
        score_path = os.path.join(trial_dir, f"fold_{fold_num}", "valid_scores.json")
        if not os.path.isfile(score_path):
            raise FileNotFoundError(
                f"Manual hyperopt trial {trial_dir} is missing validation scores at "
                f"{score_path}. Test scores cannot be used for hyperparameter selection."
            )
        with open(score_path) as f:
            scores = json.load(f)
        if metric not in scores:
            raise ValueError(
                f'Manual hyperopt trial {trial_dir} has no validation metric "{metric}" '
                f'in {score_path}.'
            )
        fold_scores.append(scores[metric])

    score_array = np.asarray(fold_scores, dtype=float)
    if np.any(np.isinf(score_array)):
        raise ValueError(
            f'Manual hyperopt trial {trial_dir} produced a non-finite '
            f'validation {metric} score.'
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        average_fold_scores = multitask_mean(
            scores=score_array,
            metric=metric,
            axis=1,
            ignore_nan_metrics=trial_args.get("ignore_nan_metrics", False),
        )
        mean_score = float(np.mean(average_fold_scores))
        std_score = float(np.std(average_fold_scores))
    if not np.isfinite(mean_score) or not np.isfinite(std_score):
        raise ValueError(
            f'Manual hyperopt trial {trial_dir} produced a non-finite '
            f'validation {metric} score.'
        )
    return mean_score, std_score


def build_search_space(search_parameters: List[str], train_epochs: int = None) -> dict:
    """
    Builds the parameter space to be searched with hyperopt trials.

    :param search_parameters: A list of parameters to be included in the search space.
    :param train_epochs: The total number of epochs to be used in training.
    :return: A dictionary keyed by the parameter names of hyperopt search functions.
    """
    available_spaces = {
        "activation": hp.choice(
            "activation", options=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"]
        ),
        "aggregation": hp.choice("aggregation", options=["mean", "sum", "norm"]),
        "aggregation_norm": hp.quniform("aggregation_norm", low=1, high=200, q=1),
        "batch_size": hp.quniform("batch_size", low=5, high=200, q=5),
        "depth": hp.quniform("depth", low=2, high=6, q=1),
        "dropout": hp.quniform("dropout", low=0.0, high=0.4, q=0.05),
        "ffn_hidden_size": hp.quniform("ffn_hidden_size", low=300, high=2400, q=100),
        "ffn_num_layers": hp.quniform("ffn_num_layers", low=1, high=3, q=1),
        "final_lr_ratio": hp.loguniform("final_lr_ratio", low=np.log(1e-4), high=0.),
        "hidden_size": hp.quniform("hidden_size", low=300, high=2400, q=100),
        "init_lr_ratio": hp.loguniform("init_lr_ratio", low=np.log(1e-4), high=0.),
        "linked_hidden_size": hp.quniform("linked_hidden_size", low=300, high=2400, q=100),
        "max_lr": hp.loguniform("max_lr", low=np.log(1e-6), high=np.log(1e-2)),
    }
    if "warmup_epochs" in search_parameters:
        if (
            not isinstance(train_epochs, int)
            or isinstance(train_epochs, bool)
            or train_epochs < 0
        ):
            raise ValueError(
                'A non-negative integer train_epochs is required when '
                'searching warmup_epochs.'
            )
        available_spaces["warmup_epochs"] = (
            hp.choice("warmup_epochs", [0])
            if train_epochs < 2
            else hp.quniform(
                "warmup_epochs", low=1, high=train_epochs // 2, q=1,
            )
        )
    space = {}
    for key in search_parameters:
        if key not in available_spaces:
            raise ValueError(f'Unsupported hyperparameter search key: {key!r}.')
        space[key] = available_spaces[key]

    return space


def merge_trials(trials: Trials, new_trials_data: List[Dict]) -> Trials:
    """
    Merge a hyperopt trials object with the contents of another hyperopt trials object.

    :param trials: A hyperopt trials object containing trials data, organized into hierarchical dictionaries.
    :param trials_data: The contents of a hyperopt trials object, `Trials.trials`.
    :return: A hyperopt trials object, merged from the two inputs.
    """
    if len(trials.trials) > 0:
        max_tid = max([trial["tid"] for trial in trials.trials])
        trial_keys = set(trials.vals.keys())
        for trial in trials.trials:
            new_trial_keys = set(trial["misc"]["vals"].keys())
            if trial_keys != new_trial_keys:
                raise ValueError(
                    f"Hyperopt trials with different search spaces cannot be combined. \
                        Across the loaded previous trials, the parameters {trial_keys} \
                        were included in the search space of some trials. At least one \
                        trial includes only the parameters {new_trial_keys}."
                )
    else:
        trial_keys = None
        max_tid = 0

    for trial in new_trials_data:
        new_trial_keys = set(trial["misc"]["vals"].keys())
        if trial_keys is None:
            trial_keys = new_trial_keys
        elif new_trial_keys != trial_keys:
            raise ValueError(
                f"Hyperopt trials with different search spaces cannot be combined. \
                    A new trial searching for parameters {new_trial_keys} was merged \
                    with another trial for parameters {trial_keys}"
            )
        tid = (
            trial["tid"] + max_tid + 1
        )  # trial id needs to be unique among this list of ids.
        hyperopt_trial = Trials().new_trial_docs(
            tids=[None], specs=[None], results=[None], miscs=[None]
        )
        hyperopt_trial[0] = trial
        hyperopt_trial[0]["tid"] = tid
        hyperopt_trial[0]["misc"]["tid"] = tid
        for key in hyperopt_trial[0]["misc"]["idxs"].keys():
            hyperopt_trial[0]["misc"]["idxs"][key] = [tid]
        trials.insert_trial_docs(hyperopt_trial)
        trials.refresh()
    return trials


def load_trials(dir_path: str, previous_trials: Trials = None) -> Trials:
    """
    Load in trials from each pickle file in the hyperopt checkpoint directory.
    Checkpoints are newly loaded in at each iteration to allow for parallel entries
    into the checkpoint folder by independent hyperoptimization instances.

    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :param previous_trials: Any previously generated trials objects that the loaded trials will be merged with.
    :return: A trials object containing the merged trials from all checkpoint files.
    """

    # List out all the pickle files in the hyperopt checkpoint directory
    hyperopt_checkpoint_files = sorted(
        os.path.join(dir_path, path)
        for path in os.listdir(dir_path)
        if path.endswith(".pkl")
    )

    # Load hyperopt trials object from each file
    loaded_trials = Trials()
    if previous_trials is not None:
        loaded_trials = merge_trials(
            loaded_trials, deepcopy(previous_trials.trials),
        )

    for path in hyperopt_checkpoint_files:
        with open(path, "rb") as f:
            trial = pickle.load(f)
            loaded_trials = merge_trials(
                loaded_trials, deepcopy(trial.trials),
            )

    return loaded_trials


def save_trials(
    dir_path: str, trials: Trials, hyperopt_seed: int, logger: logging.Logger = None
) -> None:
    """
    Saves hyperopt trial data as a `.pkl` file.

    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :param hyperopt_seed: The initial seed used for choosing parameters in hyperopt trials.
    :param trials: A trials object containing information on a completed hyperopt iteration.
    """
    if logger is None:
        info = print
    else:
        info = logger.info

    makedirs(dir_path)
    new_fname = f"{hyperopt_seed}.pkl"
    target_path = os.path.join(dir_path, new_fname)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=dir_path, prefix=f'.{hyperopt_seed}-', suffix='.pkl.tmp',
    )
    try:
        with os.fdopen(file_descriptor, "wb") as trial_file:
            pickle.dump(trials, trial_file)
            trial_file.flush()
            os.fsync(trial_file.fileno())
        try:
            # A hard link publishes the complete file atomically and, unlike
            # os.replace, never overwrites a concurrent worker's same-seed
            # result.
            os.link(temporary_path, target_path)
        except FileExistsError:
            info(
                f"When saving trial with unique seed {hyperopt_seed}, found "
                "that a trial with this seed already exists. This trial was "
                "not saved."
            )
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def get_hyperopt_seed(seed: int, dir_path: str) -> int:
    """
    Assigns a seed for hyperopt calculations. Each iteration will start with a different seed.

    :param seed: The initial attempted hyperopt seed.
    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :return: An integer for use as hyperopt random seed.
    """

    seed_path = os.path.join(dir_path, HYPEROPT_SEED_FILE_NAME)

    makedirs(seed_path, isfile=True)
    with open(seed_path, "a+", encoding="utf-8") as seed_file:
        # Hyperopt explicitly supports several processes sharing this
        # directory. Serialize the read-modify-write operation so two workers
        # can never reserve the same seed.
        fcntl.flock(seed_file.fileno(), fcntl.LOCK_EX)
        try:
            seed_file.seek(0)
            seed_tokens = seed_file.read().split()
            try:
                seeds = [int(token) for token in seed_tokens]
            except ValueError as error:
                raise ValueError(
                    f'Hyperopt seed file {seed_path} contains a non-integer value.'
                ) from error

            reserved = set(seeds)
            while seed in reserved:
                seed += 1
            seeds.append(seed)

            seed_file.seek(0)
            seed_file.truncate()
            seed_file.write(" ".join(map(str, seeds)) + "\n")
            seed_file.flush()
            os.fsync(seed_file.fileno())
        finally:
            fcntl.flock(seed_file.fileno(), fcntl.LOCK_UN)

    return seed


def load_manual_trials(
    manual_trials_dirs: List[str], param_keys: List[str], hyperopt_args: HyperoptArgs
) -> Trials:
    """
    Function for loading in manual training runs as trials for inclusion in hyperparameter search.
    Trials must be consistent with trials that would be generated in hyperparameter optimization.
    Parameters that are part of the search space do not have to match, but all others do.

    :param manual_trials_dirs: A list of paths to save directories for manual trials,
                              including args.json and per-fold valid_scores.json files.
    :param param_keys: A list of the parameters included in the hyperparameter optimization.
    :param hyperopt_args: The arguments for the hyperparameter optimization job.
    :return: A hyperopt trials object including all the loaded manual trials.
    """
    # Non-extensive list of arguments that need to match between the manual trials and the search space.
    matching_args = [
        ("number_of_molecules", None),
        ("aggregation", "aggregation"),
        ("num_folds", None),
        ("ensemble_size", None),
        ("max_lr", "max_lr"),
        ("init_lr", "init_lr_ratio"),
        ("final_lr", "final_lr_ratio"),
        ("activation", "activation"),
        ("metric", None),
        ("bias", None),
        ("epochs", None),
        ("explicit_h", None),
        ("adding_h", None),
        ("reaction", None),
        ("split_type", None),
        ("warmup_epochs", "warmup_epochs"),
        ("aggregation_norm", "aggregation_norm"),
        ("batch_size", "batch_size"),
        ("depth", "depth"),
        ("dropout", "dropout"),
        ("ffn_num_layers", "ffn_num_layers"),
        ("dataset_type", None),
        ("multiclass_num_classes", None),
        ("features_generator", None),
        ("no_features_scaling", None),
        ("features_only", None),
        ("split_sizes", None),
    ]

    manual_trials_data = []
    for i, trial_dir in enumerate(manual_trials_dirs):
        # Extract argument data from args.json
        with open(os.path.join(trial_dir, "args.json")) as f:
            trial_args = json.load(f)

        # Select manual configurations from validation data only. Never fall
        # back to the legacy test_scores.csv artifact.
        mean_score, std_score = _load_manual_validation_scores(trial_dir, trial_args)
        loss = (1 if hyperopt_args.minimize_score else -1) * mean_score

        # Check for differences in manual trials and hyperopt space
        if "linked_hidden_size" in param_keys:
            if trial_args["hidden_size"] != trial_args["ffn_hidden_size"]:
                raise ValueError(
                    f'The manual trial in {trial_dir} has a hidden_size {trial_args["hidden_size"]} '
                    f'that does not match its ffn_hidden_size {trial_args["ffn_hidden_size"]}, as it would in hyperparameter search.'
                )
        elif "hidden_size" not in param_keys or "ffn_hidden_size" not in param_keys:
            if "hidden_size" not in param_keys:
                if getattr(hyperopt_args, "hidden_size") != trial_args["hidden_size"]:
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument hidden_size than the hyperparameter optimization search trials."
                    )
            if "ffn_hidden_size" not in param_keys:
                if (
                    getattr(hyperopt_args, "ffn_hidden_size")
                    != trial_args["ffn_hidden_size"]
                ):
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument ffn_hidden_size than the hyperparameter optimization search trials."
                    )

        for arg, space_parameter in matching_args:
            if space_parameter not in param_keys:
                if getattr(hyperopt_args, arg) != trial_args[arg]:
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument {arg} than the hyperparameter optimization search trials."
                    )

        # Construct data dict
        param_dict = {}
        vals_dict = {}
        for key in param_keys:
            if key == "init_lr_ratio":
                param_value = val_value = trial_args["init_lr"] / trial_args["max_lr"]
            elif key == "final_lr_ratio":
                param_value = val_value = trial_args["final_lr"] / trial_args["max_lr"]
            elif key == "linked_hidden_size":
                param_value = val_value = trial_args["hidden_size"]
            elif key == "aggregation":
                param_value = trial_args[key]
                val_value = ["mean", "sum", "norm"].index(param_value)
            elif key == "activation":
                param_value = trial_args[key]
                val_value = ["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"].index(
                    param_value
                )
            else:
                param_value = val_value = trial_args[key]
            param_dict[key] = param_value
            vals_dict[key] = [val_value]
        idxs_dict = {key: [i] for key in param_keys}
        results_dict = {
            "loss": loss,
            "status": "ok",
            "mean_score": mean_score,
            "std_score": std_score,
            "hyperparams": param_dict,
            "num_params": 0,
            "seed": -(i + 1),
        }
        misc_dict = {
            "tid": i,
            "cmd": ("domain_attachment", "FMinIter_Domain"),
            "workdir": None,
            "idxs": idxs_dict,
            "vals": vals_dict,
        }
        trial_data = {
            "state": 2,
            "tid": i,
            "spec": None,
            "result": results_dict,
            "misc": misc_dict,
            "exp_key": None,
            "owner": None,
            "version": 0,
            "book_time": None,
            "refresh_time": None,
        }
        manual_trials_data.append(trial_data)

    trials = Trials()
    trials = merge_trials(trials=trials, new_trials_data=manual_trials_data)
    return trials


def save_config(config_path: str, hyperparams_dict: dict, max_lr: float) -> None:
    """
    Saves the hyperparameters for the best trial to a config json file.

    :param config_path: File path for the config json file.
    :param hyperparams_dict: A dictionary of hyperparameters found during the search.
    :param max_lr: The maximum learning rate value, to be used if not a search parameter.
    """
    makedirs(config_path, isfile=True)

    save_dict = {}

    for key in hyperparams_dict:
        if key == "linked_hidden_size":
            save_dict["hidden_size"] = hyperparams_dict["linked_hidden_size"]
            save_dict["ffn_hidden_size"] = hyperparams_dict["linked_hidden_size"]
        elif key == "init_lr_ratio":
            if "max_lr" not in hyperparams_dict:
                save_dict["init_lr"] = hyperparams_dict[key] * max_lr
            else:
                save_dict["init_lr"] = (
                    hyperparams_dict[key] * hyperparams_dict["max_lr"]
                )
        elif key == "final_lr_ratio":
            if "max_lr" not in hyperparams_dict:
                save_dict["final_lr"] = hyperparams_dict[key] * max_lr
            else:
                save_dict["final_lr"] = (
                    hyperparams_dict[key] * hyperparams_dict["max_lr"]
                )
        else:
            save_dict[key] = hyperparams_dict[key]

    with open(config_path, "w") as f:
        json.dump(save_dict, f, indent=4, sort_keys=True)
