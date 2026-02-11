"""ETL to pre-process evaluation datasets.

This ETL orchestrates the required steps to prepare data for model fine-tuning, including
normalization, special tags removal and masking. The result is stored as parquet.

Additionally, a subprocess to generate diffs of a random sample for manual verification is also
included. These diffs are stored in HTML format.
"""

import argparse
import datetime
import logging
import os
import sys
import time

import pandas as pd

sys.path.append("../../")
from utils.preprocessors import evaluation_pipelines
from utils.text_diffs import clean_folder, save_diff


DATASETS_BASE_PATH = "../../datasets/evaluation"
DIFFS_BASE_PATH = "./diffs"
DIFFS_SAMPLE_SIZE = 100
NEW_DATASETS_BASE_PATH = "./datasets"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""

    parser = argparse.ArgumentParser(description="ETL to pre-process the evaluation datasets.")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to pre-process"
    )
    parser.add_argument(
        "--variant", type=str, help="Dataset variant to be pre-processed"
    )
    return parser.parse_args()


def main(params):
    """Main function to run the ETL process"""

    logger.info("RUNING THE ETL TO PRE-PROCESS THE EVALUATION DATASETS...")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    try:
        dataset_name = params.dataset_name + (("_" + params.variant) if params.variant is not None else "")
        logger.info("Loading the %s dataset...", dataset_name)
        print(f"{DATASETS_BASE_PATH}/{params.dataset_name}/{dataset_name}.parquet")
        df = pd.read_parquet(
            f"{DATASETS_BASE_PATH}/{params.dataset_name}/{dataset_name}.parquet"
        )
        logger.info("Number of texts in the dataset: %d", df.shape[0])
    except FileNotFoundError:
        logger.error("Dataset %s does not exist!", dataset_name)
        sys.exit(1)

    logger.info("Applying pre-processing pipeline...")
    pipeline = evaluation_pipelines[params.dataset_name]
    df["text_clean"] = pipeline.transform(df["text"])

    logger.info("Extracting %d diffs...", DIFFS_SAMPLE_SIZE)
    diffs_path = os.path.join(DIFFS_BASE_PATH, dataset_name)
    sample_df = df.sample(DIFFS_SAMPLE_SIZE)
    clean_folder(diffs_path)
    save_diff(diffs_path, sample_df["text"], sample_df["text_clean"], sample_df["id"].astype(str))

    logger.info("Verifying completness on pre-processed texts...")
    empty_texts = (df["text"].isnull() | df["text"] == "").sum()
    empty_clean_texts = (df["text_clean"].isnull() | df["text_clean"] == "").sum()
    logger.info("Empty texts before pre-processing: %.2f%%", (empty_texts / df.shape[0]) * 100)
    logger.info("Empty texts after pre-processing: %.2f%%", (empty_clean_texts / df.shape[0]) * 100)

    logger.info("Persisting the pre-processed dataset...")
    lbl_colums = [c for c in df.columns if "label" in c]
    df = df[["id", "text_clean"] + lbl_colums].rename(columns={"text_clean": "text"})  # type: ignore
    os.makedirs(NEW_DATASETS_BASE_PATH, exist_ok=True)
    df.to_parquet(os.path.join(NEW_DATASETS_BASE_PATH, f"{dataset_name}.parquet"))


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
