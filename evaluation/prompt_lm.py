"""Script to prompt a language model through API for a supervised task.

This script implements the required steps to prompt a language model from different providers to
evaluate a supervised task. For both, OpenAI and Antropic, the process is done in batch mode.
"""

import argparse
import datetime
from glob import glob
import io
import json
import logging
import os
import sys
import time
from xml.sax import handler

from anthropic import Anthropic
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import pandas as pd
import numpy as np

sys.path.append("../")
from config.label_matchers import matchers
from utils.evaluation import (
    _convert_arrays,
    compute_metrics_sklearn,
    compute_metrics_multilabel_sklearn,
    compute_metrics_regression_sklearn,
)


DATASETS_BASE_PATH = "../preprocessing/evaluation/datasets"
PROMPT_TEMPLATES_BASE_PATH = "../assets/prompt_templates"
SHOTS_BASE_PATH = "./shots"
SPLITS_BASE_PATH = "./splits"
RESULTS_BASE_PATH = "./results/prompting"
UNKNOWN_LABEL = -2
UNKNOWN_INT_VALUE = 0


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Script to prompt a language model through API for a supervised task."
    )

    parser.add_argument(
        "--batch_ids", type=str, help="Calculate results based on the already processed batch IDs"
    )
    parser.add_argument(
        "--batch_sleep", type=int, default=10,
        help="Time in seconds to wait for creating new mini-batches and checking batch completion"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--model_id", type=str, default="gpt-4o-2024-08-06",
        choices=["gpt-4o-2024-08-06", "claude-3-7-sonnet-20250219"],
        help="Name of the model to use"
    )
    parser.add_argument(
        "--num_mini_batches", type=int, default=1,
        help="Number of mini-batches to split the requests"
    )
    parser.add_argument(
        "--shots_strategy", type=str, default="zero",
        choices=["zero-shot", "3-shot"],
        help="Prompting strategy (number of shots to use)"
    )
    parser.add_argument(
        "--task_type", type=str, default="classification",
        choices=["classification", "multilabel", "regression"],
        help="Task type to solve"
    )

    return parser.parse_args()


class OpenAIBatch:
    """..."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.requests = []
        self.batch = None
        self.status = None

        self.client = OpenAI()

    def prepare_requests(self, prompts):
        """..."""

        for prompt in prompts:
            self.requests.append({
                "custom_id": prompt["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "temperature": 0,
                    "max_tokens": 50,
                    "messages": [
                        {"role": "user", "content": prompt["content"]}
                    ]
                }
            })

    def create_batch(self):
        """..."""

        # Creating a string with the JSON lines
        jsonl = "\n".join(json.dumps(request) for request in self.requests)
        jsonl = io.BytesIO(jsonl.encode("UTF-8"))

        batch_input_file = self.client.files.create(  # type: ignore
            file=jsonl,
            purpose="batch"
        )

        self.batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info("Batch ID: %s", self.batch.id)
        self.status = "sent"

        store_batch(self.batch.id, self.requests)

    def check_status(self, batch_id=None):
        """..."""
        if batch_id is not None:
            self.batch = self.client.batches.retrieve(batch_id)  # type: ignore
        else:
            self.batch = self.client.batches.retrieve(self.batch.id)  # type: ignore
        logger.info(self.batch.request_counts.__str__)
        self.status = self.batch.status

    def retrieve_results(self):
        """..."""
        response = self.client.files.content(self.batch.output_file_id)  # type: ignore
        response = response.text
        response = response.split("\n")[:-1]
        response = [json.loads(r) for r in response]

        return pd.DataFrame([{
            "id": resp["custom_id"],
            "prediction": resp["response"]["body"]["choices"][0]["message"]["content"]
        } for resp in response]).set_index("id")


class AntropicBatch:
    """..."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.requests = []
        self.batch = None
        self.status = None

        self.client = Anthropic()

    def prepare_requests(self, prompts):
        """..."""

        for prompt in prompts:
            self.requests.append({
                "custom_id": prompt["custom_id"],
                "params": {
                    "model": self.model_id,
                    "temperature": 0,
                    "max_tokens": 50,
                    "messages": [
                        {"role": "user", "content": prompt["content"]}
                    ]
                }
            })

    def create_batch(self):
        """..."""

        self.batch = self.client.messages.batches.create(
            requests=self.requests
        )
        logger.info("Batch ID: %s", self.batch.id)
        self.status = "sent"

        store_batch(self.batch.id, self.requests)

    def check_status(self, batch_id=None):
        """..."""
        if batch_id is not None:
            self.batch = self.client.messages.batches.retrieve(batch_id)  # type: ignore
        else:
            self.batch = self.client.messages.batches.retrieve(self.batch.id)  # type: ignore
        logger.info(self.batch.request_counts.__str__)
        self.status = self.batch.processing_status

    def retrieve_results(self):
        """..."""
        response = self.client.messages.batches.results(self.batch.id)  # type: ignore

        return pd.DataFrame([{
            "id": resp.custom_id,
            "prediction": resp.result.message.content[0].text  # type: ignore
        } for resp in response]).set_index("id")


def text_to_label(text, matcher_inv):
    """Convert a text to a numerical label"""
    text = text.lower().strip()

    for k, v in matcher_inv.items():
        if k.lower() in text:
            return v

    # logger.warning("Prediction unknown: %s", text)
    return UNKNOWN_LABEL  # This value must not exist in matcher_inv


def text_to_labels(row, labels):
    """Convert a text separated by semicolon to a one-hot encoded labels"""
    onehot = []
    text = row["prediction"]

    texts = [t.lower().strip() for t in text.split(",")]
    for label in labels:
        if label.lower() in texts:
            onehot.append(1)
        else:
            onehot.append(0)

    return onehot


def text_to_int(text):
    """Convert a text to a integer"""
    try:
        return int(text)
    except ValueError:
        return UNKNOWN_INT_VALUE


def store_batch(batch_id, requests):
    """..."""
    path = os.path.join(RESULTS_BASE_PATH, "tmp")

    # Creating the folder to store the results
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"{batch_id}.json"), "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO PROMPT A LANGUAGE MODEL THROUGH API FOR A SUPERVISED TASK")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = os.path.join(
        RESULTS_BASE_PATH, params.shots_strategy, params.dataset_name, params.model_id
    )

    matcher = matchers[params.dataset_name]
    if params.task_type == "classification":
        matcher_inv = {v: k for k, v in matcher.items()}
        # Sort descending by key length
        matcher_inv = dict(sorted(matcher_inv.items(), key=lambda item: len(item[0]), reverse=True))

    # Setting configurations by provider
    if "gpt" in params.model_id:
        batch_handlers = [OpenAIBatch(params.model_id) for _ in range(params.num_mini_batches)]
    elif "claude" in params.model_id:
        batch_handlers = [AntropicBatch(params.model_id) for _ in range(params.num_mini_batches)]
    else:
        logger.error("Provider not configured for model_id %s", params.model_id)
        sys.exit(1)

    # Creating the folder to store the results
    os.makedirs(results_base_path, exist_ok=True)

    try:
        logger.info("Loading the %s dataset...", params.dataset_name)
        data_df = pd.read_parquet(
            os.path.join(DATASETS_BASE_PATH, f"{params.dataset_name}.parquet")
        )

        splits_filename = f"{params.dataset_name}.*.*.csv"
        logger.info("Loading the splits file named %s...", splits_filename)
        splits_df = pd.read_csv(glob(os.path.join(SPLITS_BASE_PATH, splits_filename))[0])
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    logger.info("Loading the data for testing...")
    test_df = data_df.loc[splits_df["fold_1"] == 2]
    logger.info("Number of instances for testing: %d", test_df.shape[0])

    if params.shots_strategy == "3-shots":
        logger.info("Generating shots...")
        shots_df = data_df.loc[splits_df["fold_1"] != 2]
        shot_ids_df = pd.read_csv(
            os.path.join(SHOTS_BASE_PATH, f"{params.dataset_name}.csv"), index_col=0
        )

        test_df = pd.concat([test_df, shot_ids_df], axis=1)

        labels = [c for c in test_df.columns if "label_" in c]

        for i in range(3):  # This range has to be dynamic if shots are different than 3
            shot_ids = test_df[str(i)].values
            shot_texts, shot_labels = [], []
            for shot_id in shot_ids:
                shot = shots_df.loc[shots_df["id"] == shot_id].iloc[0]
                shot_texts.append(shot["text"])
                if params.task_type == "classification":
                    shot_labels.append(matcher[shot["label"]])
                elif params.task_type == "multilabel":
                    labels_str = ",".join(
                        [
                            lbl.replace("label_", "")
                            for lbl in shot[labels].loc[shot[labels] == 1].index
                        ]
                    )
                    shot_labels.append(labels_str)
                elif params.task_type == "regression":
                    shot_labels.append(str(shot["label"]))

            test_df[f"shot_{i+1}"] = shot_texts
            test_df[f"label_{i+1}"] = shot_labels

    if params.batch_ids is None:

        try:
            prompt_filename = os.path.join(
                PROMPT_TEMPLATES_BASE_PATH, f"{params.dataset_name}.{params.shots_strategy}.txt"
            )
            logger.info("Loading the prompt template named %s...", prompt_filename)
            prompt_template = PromptTemplate.from_file(prompt_filename)
        except FileNotFoundError as e:
            logger.error(e)
            sys.exit(1)

        logger.info("Preparing the requests...")
        splits = np.array_split(test_df, params.num_mini_batches)
        for i, (split, batch_handler) in enumerate(zip(splits, batch_handlers), start=1):

            prompts = []
            for ix, row in split.iterrows():  # type: ignore
                if params.shots_strategy == "3-shots":
                    prompt = prompt_template.format(
                        text=row["text"],
                        shot_1=row["shot_1"], label_1=row["label_1"],
                        shot_2=row["shot_2"], label_2=row["label_2"],
                        shot_3=row["shot_3"], label_3=row["label_3"],
                    )
                else:
                    prompt = prompt_template.format(text=row["text"])

                prompts.append({
                    "custom_id": str(ix),
                    "content": prompt
                })
            batch_handler.prepare_requests(prompts)

            logger.info("Creating the mini-batch %d of %d...", i, len(splits))
            batch_handler.create_batch()

            if i < len(splits):
                logger.info(
                    "Waiting %d seconds to launch the next mini-batch...", params.batch_sleep
                )
                time.sleep(params.batch_sleep)

        logger.info("Reporting mini-batch IDs...")
        batch_ids = [h.batch.id for h in batch_handlers]  # type: ignore
        logger.info(",".join(batch_ids))

        all_batches_finished = False
        while not all_batches_finished:
            num_completed = 0
            for batch_handler in batch_handlers:
                batch_handler.check_status()
                if batch_handler.status in ["completed", "ended"]:
                    num_completed += 1
            logger.info("%d of %d mini batches completed", num_completed, params.num_mini_batches)
            all_batches_finished = num_completed == params.num_mini_batches

            if num_completed < params.num_mini_batches:
                logger.info("Waiting %d seconds to check status...", params.batch_sleep)
                time.sleep(params.batch_sleep)

    else:

        batch_ids = params.batch_ids.split(",")

        if len(batch_ids) != params.num_mini_batches:
            logger.error("Number of batch IDs passed as parameter does not match")
            sys.exit(1)

        all_batches_finished = False
        while not all_batches_finished:
            num_completed = 0
            for batch_id, batch_handler in zip(batch_ids, batch_handlers):
                batch_handler.check_status(batch_id)
                if batch_handler.status in ["completed", "ended"]:
                    num_completed += 1
            logger.info("%d of %d mini batches completed", num_completed, params.num_mini_batches)
            all_batches_finished = num_completed == params.num_mini_batches

            if num_completed < params.num_mini_batches:
                logger.info("Waiting %d seconds to check status...", params.batch_sleep)
                time.sleep(params.batch_sleep)

    logger.info("Retrieving results...")
    responses = []
    for batch_handler in batch_handlers:
        response_df = batch_handler.retrieve_results()

        if params.task_type == "classification":
            response_df["prediction"] = response_df["prediction"].apply(
                lambda x: text_to_label(x, matcher_inv)
            )
        elif params.task_type == "multilabel":
            response_df[[f"pred_{v}" for v in matcher]] = response_df.apply(
                lambda x: text_to_labels(x, matcher),
                axis=1,
                result_type="expand"
            )
        elif params.task_type == "regression":
            response_df["prediction"] = response_df["prediction"].apply(text_to_int)

        responses.append(response_df)
    responses_df = pd.concat(responses)

    # Making sure both structures have the same order
    test_df.sort_index(inplace=True)
    responses_df.index = responses_df.index.astype(int)
    responses_df.sort_index(inplace=True)

    if test_df.shape[0] != responses_df.shape[0]:
        logger.warning(
            "Test and response datasets have inconsistent number of instances. Difference: %d",
            test_df.shape[0] - responses_df.shape[0]
        )
        common_indexes = test_df.index.intersection(responses_df.index)
        test_df = test_df.loc[common_indexes]
        responses_df = responses_df.loc[common_indexes]
        logger.warning(
            "Calculating error metrics on instances: %d",
            len(common_indexes)
        )

    logger.info("Reporting error metrics...")
    if params.task_type == "classification":
        num_unkws = responses_df["prediction"].value_counts().get(UNKNOWN_LABEL, 0)
        handle_unkws = num_unkws > 0
        logger.warning("Predictions unknown found: %d", num_unkws)
        results = compute_metrics_sklearn(
            test_df["label"], responses_df["prediction"], handle_unkws=handle_unkws
        )
        results["unknown_predictions"] = num_unkws
    elif params.task_type == "multilabel":
        results = compute_metrics_multilabel_sklearn(
            test_df[[f"label_{v}" for v in matcher]], responses_df[[f"pred_{v}" for v in matcher]]
        )
    elif params.task_type == "regression":
        num_unkws = responses_df["prediction"].value_counts().get(UNKNOWN_INT_VALUE, 0)
        logger.warning("Predictions unknown found: %d", num_unkws)
        results = compute_metrics_regression_sklearn(test_df["label"], responses_df["prediction"])

    with open(os.path.join(results_base_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(_convert_arrays(results), f, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
