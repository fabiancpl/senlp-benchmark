"""ETL pre-processing pipelines per dataset.

This module specifies the pipelines to be applied to the different datasets and their respective
entities. The pipelines need to be defined using the Scikit-Learn Pipeline interface.
"""

import sys

import evaluate
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
import torch
import torch.nn.functional as F

sys.path.append("../")
from utils.metrics.smape.smape import symmetric_mean_absolute_percentage_error


def compute_metrics_sklearn(y_true, y_pred, handle_unkws=False):
    """Calculate different classification scores for binary and multi-class classification using
    vanilla Scikit-Learn"""

    n_labels = y_true.nunique()

    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    for average in ["micro", "macro", None]:
        adj = 1
        if (average == "macro") and handle_unkws:
            adj = (n_labels + 1) / n_labels

        results[f"f1_{average}"] = f1_score(y_true, y_pred, average=average) * adj
        results[f"precision_{average}"] = precision_score(y_true, y_pred, average=average) * adj
        results[f"recall_{average}"] = recall_score(y_true, y_pred, average=average) * adj

    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()
    }


def compute_metrics_multilabel_sklearn(y_true, y_pred):
    """Calculate different classification scores for multilabel classification using Scikit-Learn"""

    results = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)  # Strict match across all labels
    results["confusion_matrices"] = [
        confusion_matrix(
            y_true[col1], y_pred[col2]
        ).tolist() for col1, col2 in zip(y_true.columns, y_pred.columns)
    ]

    for average in ["micro", "macro", None]:
        results[f"f1_{average}"] = f1_score(y_true, y_pred, average=average)
        results[f"precision_{average}"] = precision_score(y_true, y_pred, average=average)
        results[f"recall_{average}"] = recall_score(y_true, y_pred, average=average)

    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()
    }


def compute_metrics_regression_sklearn(y_true, y_pred):
    """Calculate different regression scores using vanilla Scikit-Learn"""

    results = {}
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    results["smape"] = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    return results


def compute_metrics_hf(eval_pred):
    """Calculate different classification scores for binary and multi-class classification using
    Hugging Face's evaluate library"""

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]  # Extract the actual logits (usually the first element)

    predictions = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("../utils/metrics/accuracy")
    conf_matrix = evaluate.load("../utils/metrics/confusion_matrix")
    metrics = evaluate.combine(
        ["../utils/metrics/f1", "../utils/metrics/precision", "../utils/metrics/recall"]
    )

    results = {
        **accuracy.compute(predictions=predictions, references=labels),  # type: ignore
        **conf_matrix.compute(predictions=predictions, references=labels)  # type: ignore
    }

    for average in ["micro", "macro", None]:
        res = metrics.compute(predictions=predictions, references=labels, average=average)
        res = {f"{k}_{average}": v for k, v in res.items()}
        results = {**results, **res}

    return _convert_arrays(results)


def compute_metrics_multilabel_hf(eval_pred):
    """Calculate different classification scores for multilabel classification using Hugging Face's
    evaluate library"""

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]  # Extract the actual logits (usually the first element)

    # Apply sigmoid to get probabilities for multi-label classification
    probabilities = 1 / (1 + np.exp(-logits))
    threshold = 0.5
    predictions = (probabilities > threshold).astype(int)
    labels = labels.astype(int)

    accuracy = evaluate.load("../utils/metrics/accuracy", "multilabel")
    f1 = evaluate.load("../utils/metrics/f1", "multilabel")
    precision = evaluate.load("../utils/metrics/precision", "multilabel")
    recall = evaluate.load("../utils/metrics/recall", "multilabel")

    results = {
        **accuracy.compute(predictions=predictions, references=labels),  # type: ignore
    }

    results = {}
    results["accuracy"] = accuracy.compute(predictions=predictions, references=labels)

    for average in ["micro", "macro", None]:
        f1_res = f1.compute(predictions=predictions, references=labels, average=average)
        f1_res = {f"{k}_{average}": v for k, v in f1_res.items()}  # type: ignore

        precision_res = precision.compute(
            predictions=predictions, references=labels, average=average
        )
        precision_res = {f"{k}_{average}": v for k, v in precision_res.items()}  # type: ignore

        recall_res = recall.compute(predictions=predictions, references=labels, average=average)
        recall_res = {f"{k}_{average}": v for k, v in recall_res.items()}  # type: ignore

        results = {**results, **f1_res, **precision_res, **recall_res}

    return _convert_arrays(results)


def compute_metrics_regression_hf(eval_pred):
    """Calculate different regression scores using Hugiing Face's evaluate library"""

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract the actual logits (usually the first element)

    mae = evaluate.load("../utils/metrics/mae")
    mse = evaluate.load("../utils/metrics/mse")
    mape = evaluate.load("../utils/metrics/mape")
    smape = evaluate.load("../utils/metrics/smape")

    results = {
        "mae": mae.compute(predictions=predictions, references=labels)["mae"],  # type: ignore
        "rmse": mse.compute(
            predictions=predictions, references=labels, squared=True
        )["mse"],  # type: ignore
        "mape": mape.compute(predictions=predictions, references=labels)["mape"],  # type: ignore
        "smape": smape.compute(predictions=predictions, references=labels)["smape"]  # type: ignore
    }

    return _convert_arrays(results)


def compute_metrics_ner_hf(eval_preds, label_names):
    """Calculate different NER scores using Hugiing Face's evaluate library"""

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[lbl] for lbl in label if lbl != -100] for label in labels]
    true_predictions = [
        [label_names[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metrics = evaluate.load("../utils/metrics/seqeval")

    results = metrics.compute(predictions=true_predictions, references=true_labels)

    return _convert_arrays(results)


def compute_metrics_mlm_hf(eval_preds):
    """Calculate different MLM scores using Hugiing Face's evaluate library"""

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1).reshape(-1)
    labels = labels.reshape(-1)

    # Create mask for masked token positions (labels != -100)
    mask = labels != -100
    masked_preds = predictions[mask]
    masked_labels = labels[mask]

    accuracy = evaluate.load("../utils/metrics/accuracy")

    results = {}
    results["accuracy"] = accuracy.compute(
        predictions=masked_preds, references=masked_labels
    )["accuracy"]  # type: ignore

    # Perplexity via manual cross-entropy
    if masked_labels.size > 0:
        # Rebuild logits for masked positions
        vocab_size = logits.shape[-1]
        logits_2d = logits.reshape(-1, vocab_size)
        masked_logits = torch.tensor(logits_2d[mask], dtype=torch.float32)
        masked_labels_tensor = torch.tensor(masked_labels, dtype=torch.long)
        loss = F.cross_entropy(masked_logits, masked_labels_tensor, reduction="mean")
        results["perplexity"] = float(torch.exp(loss))
    else:
        results["perplexity"] = float("inf")

    return _convert_arrays(results)


def _convert_arrays(data):
    """Make sure the results dictionary is serializable"""
    if isinstance(data, dict):
        return {k: _convert_arrays(v) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.int64):  # type: ignore
        return data.item()
    return data
