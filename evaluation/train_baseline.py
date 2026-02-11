"""Script to train a baseline model for a supervised task.

This script implements the required steps to build a baseline model including training, model
selection, and evaluation. Two frameworks are supported: XGBoost (Scikit-Learn API) and FastText.
"""

import argparse
import datetime
from glob import glob
from joblib import (
    Parallel,
    delayed,
    parallel_backend,
)
import json
import logging
import os
import re
import sys
import time
from tqdm import tqdm

import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

sys.path.append("../")
from utils.evaluation import (
    _convert_arrays,
    compute_metrics_sklearn,
    compute_metrics_multilabel_sklearn,
    compute_metrics_regression_sklearn,
)


DATASETS_BASE_PATH = "../preprocessing/evaluation/datasets"
EMBEDDING_MODEL = "facebook/fasttext-en-vectors"
RESULTS_BASE_PATH = "./results/baselines"
SPLITS_BASE_PATH = "./splits"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script to train a baseline model for a supervised task."
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--model_type", type=str, default="xgboost", choices=["xgboost", "fasttext"],
        help="Baseline model type. xgboost means TF-IDF+XBoost using the Scikit-Learn API"
    )
    parser.add_argument(
        "--task_type", type=str, default="classification",
        choices=["classification", "multilabel", "regression"],
        help="Task type to solve"
    )

    return parser.parse_args()


def get_labels(predictions):
    """Extract a simple prediction list"""
    return [int(label[0].replace("__label__", "")) for label in predictions[0]]


def get_labels_multilabel(predictions, labels):
    """Extract a prediction matrix (for multilabel)"""

    thr = 0.5  # Decision threshold
    results = []

    for _, (preds, probs) in enumerate(zip(predictions[0], predictions[1])):
        # Convert FastText labels from "__label__labelname" to "label_labelname"
        # Filter by decision threshold
        filtered_preds = {
            label.replace("__label__", "label_") for j, label in enumerate(preds) if probs[j] >= thr
        }

        # Convert to one-hot representation
        onehot = [1 if label in filtered_preds else 0 for label in labels]

        # Store results if needed later
        results.append(onehot)

    return results


def clean_quotes(filename):
    """Remove quotes from temp file"""
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    # Remove single and double quotes
    content = content.replace('"', '')

    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """Text vectorizer based on FastText embedding model"""

    def __init__(self, model_repo=EMBEDDING_MODEL, model_filename="model.bin"):
        """Custom transformer to convert text into FastText word vector averages."""
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.model = None
        self.model_dim = None
        self.word_vectors = {}

    def _load_model(self):
        """Loads the FastText model from Hugging Face (only once)."""
        if self.model is None:
            print("Downloading model...")
            model_path = hf_hub_download(repo_id=self.model_repo, filename=self.model_filename)
            print("Loading model...")
            self.model = fasttext.load_model(model_path)
            self.model_dim = self.model.get_dimension()
            print("Preparing word vectors...")
            self.word_vectors = {word: self.model[word] for word in self.model.words}

    def fit(self, X, y=None):
        """No fitting required, returns self."""
        self._load_model()
        return self

    def transform(self, X):
        """Transforms a list of texts into their FastText vector representations."""
        self._load_model()
        print("Vectorizing texts...")
        with parallel_backend("threading"):
            return np.array(Parallel(n_jobs=-1)(delayed(self._text_to_vector)(text) for text in X))

    def _text_to_vector(self, text):
        """Tokenizes and converts a text input into a FastText vector."""
        words = re.findall(r"(?u)\b\w\w+\b", text.lower())
        word_vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]

        if not word_vectors:
            return np.zeros(self.model_dim if self.model_dim is not None else 300)

        return np.mean(word_vectors, axis=0)


def run_experiment_sklearn(train_df, test_df, task_type, labels, max_depth=6, min_child_weight=1):
    """Train a XGBoost model with the given hyper-parameters using the Scikit-Learn interface"""

    if task_type == "multiclass":
        model = XGBClassifier(
            max_depth=max_depth, min_child_weight=min_child_weight,
            objective="multi:softmax", eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )
    elif task_type == "multilabel":
        model = MultiOutputClassifier(XGBClassifier(
            max_depth=max_depth, min_child_weight=min_child_weight,
            random_state=42, n_jobs=-1
        ))
    elif task_type == "regression":
        model = XGBRegressor(
            max_depth=max_depth, min_child_weight=min_child_weight,
            random_state=42, n_jobs=-1
        )
    else:
        model = XGBClassifier(
            max_depth=max_depth, min_child_weight=min_child_weight,
            random_state=42, n_jobs=-1
        )
    pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("model", model)
        ])

    if task_type == "multilabel":
        pipeline.fit(train_df["text"], train_df[labels])
    else:
        pipeline.fit(train_df["text"], train_df["label"])
    train_preds = pipeline.predict(train_df["text"])
    test_preds = pipeline.predict(test_df["text"])

    return train_preds, test_preds


def run_experiment_fasttext(train_df, test_df, task_type, labels, lr=0.1, epoch=5):
    """Train a FastText model with the given hyper-parameters"""
    tmp_filename = "./tmp/train.txt"

    train_df[["label_mod", "text"]].to_csv(
        tmp_filename, sep=" ", index=False, header=None  # type: ignore
    )

    if task_type == "multilabel":
        # Workaround to format properly multiple labels per instance
        clean_quotes(tmp_filename)

        model = fasttext.train_supervised(tmp_filename, lr=lr, epoch=epoch, loss="ova")

        train_preds = get_labels_multilabel(
            model.predict(train_df["text"].tolist(), k=len(labels)), labels
        )
        test_preds = get_labels_multilabel(
            model.predict(test_df["text"].tolist(), k=len(labels)), labels
        )
    else:
        model = fasttext.train_supervised(tmp_filename, lr=lr, epoch=epoch)

        train_preds = get_labels(model.predict(train_df["text"].tolist()))
        test_preds = get_labels(model.predict(test_df["text"].tolist()))

    return train_preds, test_preds


def run_experiment_fasttext_regression(train_df, test_df):
    """Train a linear model using FastText as vectorizer"""

    pipeline = Pipeline([
            ('vectorizer', FastTextVectorizer()),
            ('model', LinearRegression())
        ])

    pipeline.fit(train_df["text"], train_df["label"])

    train_preds = pipeline.predict(train_df["text"])
    test_preds = pipeline.predict(test_df["text"])

    return train_preds, test_preds


def run_experiment(train_df, test_df, model_type, compute_metrics, task_type, labels=None, **kwargs):
    """Train a model and compute evaluation metrics"""

    if model_type == "fasttext":
        if task_type == "regression":
            train_preds, test_preds = run_experiment_fasttext_regression(train_df, test_df)
        else:
            train_preds, test_preds = run_experiment_fasttext(
                train_df, test_df, task_type, labels,
                lr=kwargs["lr"], epoch=kwargs["epoch"]
            )
    else:
        train_preds, test_preds = run_experiment_sklearn(
            train_df, test_df, task_type, labels,
            max_depth=kwargs["max_depth"], min_child_weight=kwargs["min_child_weight"]
        )

    if task_type == "multilabel":
        train_results = compute_metrics(train_df[labels], pd.DataFrame(train_preds))
        test_results = compute_metrics(test_df[labels], pd.DataFrame(test_preds))
    else:
        train_results = compute_metrics(train_df["label"], train_preds)
        test_results = compute_metrics(test_df["label"], test_preds)

    return train_results, test_results


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO TRAIN A BASELINE MODEL FOR A SUPERVISED TASK")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = os.path.join(RESULTS_BASE_PATH, params.model_type, params.dataset_name)

    # Creating the folder to store the results
    os.makedirs(results_base_path, exist_ok=True)

    # Definitions based on task type
    if params.task_type == "classification":
        metric_for_best_model = "f1_macro"
        compute_metrics = compute_metrics_sklearn
    elif params.task_type == "multilabel":
        metric_for_best_model = "f1_macro"
        compute_metrics = compute_metrics_multilabel_sklearn
    elif params.task_type == "regression":
        metric_for_best_model = "smape"
        compute_metrics = compute_metrics_regression_sklearn
    else:
        logger.error("Compute metrics function not implemented for %s", params.task_type)
        sys.exit(1)

    try:
        logger.info("Loading the %s dataset...", params.dataset_name)
        data_df = pd.read_parquet(
            os.path.join(DATASETS_BASE_PATH, f"{params.dataset_name}.parquet")
        )

        labels = None
        if params.task_type == "multilabel":
            labels = [c for c in data_df.columns if "label_" in c]
            num_labels = len(labels)
        elif params.task_type == "regression":
            num_labels = 1
        else:
            num_labels = data_df["label"].nunique()

        if params.model_type == "fasttext":
            # Regression task eses pre-trained word vectors, so label transforming is not required
            if params.task_type == "multilabel":
                data_df["label_mod"] = data_df[labels].apply(
                    lambda x: " ".join(
                        [k.replace("label_", "__label__") for k, v in x.items() if v == 1]
                    ), axis=1
                )
            elif params.task_type == "classification":
                data_df["label_mod"] = data_df["label"].apply(lambda x: "__label__" + str(x))
            data_df["text"] = data_df["text"].str.replace("\n", " ")

        splits_filename = f"{params.dataset_name}.*.*.csv"
        logger.info("Loading the splits file named %s...", splits_filename)
        splits_df = pd.read_csv(glob(os.path.join(SPLITS_BASE_PATH, splits_filename))[0])
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    logger.info("Splitting the data for final testing")
    train_df = data_df.loc[splits_df["fold_1"] != 2]
    test_df = data_df.loc[splits_df["fold_1"] == 2]

    logger.info("Defining experimentation grid...")
    combinations = []
    if params.model_type == "fasttext":
        if params.task_type != "regression":
            learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
            epochs = [5, 10, 15, 20]
            combinations = [
                {
                    "lr": learning_rate,
                    "epoch": epoch
                } for learning_rate in learning_rates for epoch in epochs
            ]
    else:
        max_depths = [3, 4, 5, 6, 7, 8, 9, 10]
        min_child_weights = [1, 3, 5, 7]
        combinations = [
            {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight
            } for max_depth in max_depths for min_child_weight in min_child_weights
        ]
    logger.info("Number of experiments (combinations): %d", len(combinations))

    logger.info("Running experiments...")
    best_combination = None
    if len(combinations) == 0:
        train_results, test_results = run_experiment(
            train_df, test_df, params.model_type, compute_metrics, params.task_type
        )
    else:
        values = []
        for combination in tqdm(combinations):
            if params.task_type == "multilabel":
                train_results, test_results = run_experiment(
                    train_df, test_df, params.model_type, compute_metrics, params.task_type, labels,
                    **combination
                )
            else:
                train_results, test_results = run_experiment(
                    train_df, test_df, params.model_type, compute_metrics, params.task_type,
                    **combination
                )

            values.append({**combination, **{
                "train": train_results[metric_for_best_model],
                "test": test_results[metric_for_best_model],
                "diff": abs(
                    train_results[metric_for_best_model] - test_results[metric_for_best_model]
                )
            }})

        logger.info("Getting the best combination of hyper-parameters...")
        values_df = pd.DataFrame(values)
        best_result = values_df.sort_values(by="test", ascending=False).iloc[0]

        if params.model_type == "fasttext":
            best_combination = {
                "lr": best_result["lr"].astype(float),
                "epoch": best_result["epoch"].astype(int)
            }
        else:
            best_combination = {
                "max_depth": best_result["max_depth"].astype(int),
                "min_child_weight": best_result["min_child_weight"].astype(int)
            }

        logger.info("Validating results for the best combination of hyper-parameters")
        if (params.task_type == "classification") and (num_labels == 2):
            train_results, test_results = run_experiment(
                train_df, test_df, params.model_type, compute_metrics, params.task_type,
                **best_combination
            )
        else:
            train_results, test_results = run_experiment(
                train_df, test_df, params.model_type, compute_metrics, params.task_type, labels,
                **best_combination
            )

    with open(os.path.join(results_base_path, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=4)

    with open(os.path.join(results_base_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    if best_combination:
        with open(os.path.join(results_base_path, "best_model.json"), "w", encoding="utf-8") as f:
            json.dump(_convert_arrays(best_combination), f, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
