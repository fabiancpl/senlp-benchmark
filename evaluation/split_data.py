"""ETL to split the data using cross-validation.

This ETL implements the data splitting process for cross-validation using strategies such as
Stratified k-Fold, Repeated K-Fold or Leave One Group Out, as defined by Scikit-Learn here:
https://scikit-learn.org/stable/modules/cross_validation.html. These strategies are supported for
binary and multi-class classification, and regression.

For multi-label classification and ranking tasks with multiple targets, an iterative strategy is
used as defined here: http://scikit.ml/stratification.html

For NER and MLM tasks a simple random splitting is implemented.

The resulting file is a CSV with the same number rows of the source dataset but with columns
corresponding to each fold, depending on the strategy selected. The values follow this notation:
1 - Instance used for training in the current fold
0 - Instance used for validation in the current fold
2 - Intance used for testing at the end of the cross-validation experiment

Note that instances assigned for testing are the same for all the folds.

NOTE:
- Arguments --cv_strategy (leave-one-group-out|repeated-k-fold) --repetitions and --group will be
removed in future versions.
"""

import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    LeaveOneGroupOut,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from skmultilearn.model_selection import (
    iterative_train_test_split,
    IterativeStratification
)


DATASETS_BASE_PATH = "../preprocessing/evaluation/datasets"
SPLITS_BASE_PATH = "./splits"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""

    parser = argparse.ArgumentParser(description="ETL to split the data using cross-validation.")

    parser.add_argument(
        "--cv_strategy", type=str, default="k-fold",
        choices=["k-fold", "repeated-k-fold", "leave-one-group-out", "iterative"],
        help="Splitting strategy"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to split"
    )
    parser.add_argument(
        "--group", type=str, help="Grouping column to be used when strategy is leave-one-group-out"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of folds to be used when strategy is k-fold or repeated-k-fold"
    )
    parser.add_argument(
        "--no_stratify", action="store_true",
        help="Stratification is not taken into account. Compatible only with cv_strategy k-fold"
    )
    parser.add_argument(
        "--repetitions", type=int, default=10,
        help="Number of repetitions to be used when strategy is repeated-k-fold"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Percentage of the data reserved for testing"
    )

    return parser.parse_args()


def main(params):
    """Main function to run the ETL process"""

    logger.info("RUNNING THE ETL TO SPLIT THE DATA USING CROSS-VALIDATION...")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    try:
        logger.info("Loading the %s dataset...", params.dataset_name)
        # WARNING: The cased version of the dataset is asumed as default, so it must always exist!
        df = pd.read_parquet(f"{DATASETS_BASE_PATH}/{params.dataset_name}.parquet")
    except FileNotFoundError:
        logger.error("Dataset %s does not exist!", params.dataset_name)
        sys.exit(1)

    logger.info("Reserving %.2f%% of the data for testing...", params.test_size * 100)
    target = "label"
    if params.cv_strategy == "iterative":
        target = [c for c in df.columns if "label_" in c]
        logger.info("Applying a iterative splitting strategy over %d labels...", len(target))
        train_val_df, test_df, _, _ = iterative_train_test_split(
            X=df,  # type: ignore
            y=df[target],  # type: ignore
            test_size=params.test_size
        )
    else:
        logger.info(df.columns)

        train_val_df, test_df = train_test_split(
            df,
            test_size=params.test_size,
            random_state=42,
            stratify=(df[target] if not params.no_stratify else None)
        )
    logger.info("Train-validation dataset: %d", train_val_df.shape[0])
    logger.info("Test dataset: %d", test_df.shape[0])

    logger.info(
        "Splitting the train-validation dataset using the %s strategy...", params.cv_strategy
    )
    n_folds = 0
    splits_filename = f"{params.dataset_name}.{params.cv_strategy}"
    if params.cv_strategy == "k-fold":
        if params.no_stratify:
            strategy = KFold(n_splits=params.k, shuffle=True, random_state=42)
        else:
            strategy = StratifiedKFold(n_splits=params.k, shuffle=True, random_state=42)
        n_folds = params.k
        splits_filename += f".{params.k}k"
    elif params.cv_strategy == "repeated-k-fold":
        strategy = RepeatedStratifiedKFold(
            n_splits=params.k, n_repeats=params.repetitions, random_state=42
        )
        n_folds = params.k * params.repetitions
        splits_filename += f".{params.k}k.{params.repetitions}reps"
    elif params.cv_strategy == "leave-one-group-out":
        strategy = LeaveOneGroupOut()
        n_folds = df[params.group].nunique()
        splits_filename += f".{n_folds}{params.group}"
    elif params.cv_strategy == "iterative":
        strategy = IterativeStratification(n_splits=params.k, random_state=42)
        n_folds = params.k
        splits_filename += f".{params.k}k"
    else:
        logger.error("Strategy %s not implemented yet!", params.cv_strategy)
        sys.exit(1)

    logger.info("Generating %d folds...", n_folds)
    train_val_splits_arr = np.zeros((train_val_df.shape[0], n_folds), dtype=int)
    for fold, (train_idx, _) in enumerate(
        strategy.split(
            X=train_val_df,  # type: ignore
            y=train_val_df[target],
            groups=df[params.group] if params.cv_strategy == "leave-one-group-out" else None
        )
    ):
        train_val_splits_arr[train_idx, fold] = 1
    train_val_splits_df = pd.DataFrame(
        train_val_splits_arr,
        columns=[f"fold_{i+1}" for i in range(n_folds)],
        index=train_val_df.index
    )
    test_splits_df = pd.DataFrame(
        np.full((test_df.shape[0], n_folds), 2, dtype=int),
        columns=[f"fold_{i+1}" for i in range(n_folds)],
        index=test_df.index
    )
    splits_df = pd.concat([
        train_val_splits_df,
        test_splits_df
    ]).sort_index()

    logger.info("Persisting splits as a CSV file named '%s.csv'...", splits_filename)
    os.makedirs(SPLITS_BASE_PATH, exist_ok=True)
    splits_df.to_csv(f"{SPLITS_BASE_PATH}/{splits_filename}.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
