"""Script to fine-tune a language model.

This script implements the required steps to fine-tune a language model including tokenization,
training and evaluation, depending on the cross-validation strategy selected.
"""

import argparse
import datetime
from glob import glob
import logging
import os
import sys
import time

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value
)
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    IntervalStrategy,
    pipeline,
    Trainer,
    TrainingArguments,
)

sys.path.append("../")
from config.model_directory import hf_models
from utils.evaluation import (
    compute_metrics_hf,
    compute_metrics_multilabel_hf,
    compute_metrics_regression_hf,
)


DATASETS_BASE_PATH = "../preprocessing/evaluation/datasets"
SPLITS_BASE_PATH = "./splits"
PRETRAINED_MODELS_BASE_PATH = "../pretraining/models"
PRETRAINED_TOKENIZERS_BASE_PATH = "../pretraining/tokenizers"
RESULTS_BASE_PATH = "./results/finetuning"


logger = logging.getLogger(__name__)


def handle_args():
    """Handle command line arguments"""
    parser = argparse.ArgumentParser(description="Script to fine-tune a language model.")

    # Task arguments
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--task_type", type=str, default="classification",
        choices=["classification", "multilabel", "regression", "ner", "mlm"],
        help="Task type to solve"
    )

    # Model arguments
    parser.add_argument(
        "--base_model_name", type=str, help="Name of the base model"
    )
    parser.add_argument(
        "--base_model_path", type=str, help="Local path of the base model."
    )
    parser.add_argument(
        "--max_length", type=int,
        help="Max. length for sentence truncation"
    )
    parser.add_argument(
        "--not_use_safetensors", action="store_true",
        help="Extended compatibility with old models like CodeT5+"
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true",
        help="Specify if the training must be resumed from a checkpoint located at the default path"
    )
    parser.add_argument(
        "--tokenizer_name", type=str,
        help="Name of the tokenizer. It has to exitst in model directory"
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        help="Local path of the tokenizer"
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="Experiment version"
    )

    # Hyper-parameters
    parser.add_argument(
        "--batch_size_per_gpu", type=int, required=True,
        help="Batch size for training and testing (per device)"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_accumulation_steps", type=int,
        help="Accumulate evaluation results to deal with memory error"
    )
    parser.add_argument(
        "--flash_attention", action="store_true",
        help="Specify if using flash attention 2"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, required=True,
        help="Number of update steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Use gradient checkpointing"
    )
    parser.add_argument(
        "--learning_rate", type=float, required=True, help="Initial learning rate"
    )

    # Aux: Required for compatibility with DeepSpeed
    parser.add_argument(
        "--local_rank", type=int, default=1,
        help="..."
    )

    return parser.parse_args()


def main(params):
    """Main function to run the script"""

    logger.info("RUNNING THE SCRIPT TO FINE-TUNE A LANGUAGE MODEL")
    logger.info("Arguments passed by command-line: %s", params)
    logger.info("Execution date: %s", datetime.datetime.now())

    results_base_path = os.path.join(
        RESULTS_BASE_PATH, params.dataset_name, params.version,
        params.base_model_name or params.base_model_path
    )

    # Definitions based on task type
    if params.task_type == "classification":
        problem_type = None
        compute_metrics = compute_metrics_hf
    elif params.task_type == "multilabel":
        problem_type = "multi_label_classification"
        compute_metrics = compute_metrics_multilabel_hf
    elif params.task_type == "regression":
        problem_type = "regression"
        compute_metrics = compute_metrics_regression_hf
    else:
        logger.error("Compute metrics function not implemented for %s", params.task_type)
        sys.exit(1)

    try:
        logger.info("Loading the %s dataset...", params.dataset_name)
        data_df = pd.read_parquet(os.path.join(DATASETS_BASE_PATH, f"{params.dataset_name}.parquet"))

        # Workaround to avoid T5 tokenization errors
        data_df["text"] = data_df["text"].str.replace("</s>", "")

        if params.task_type == "multilabel":
            labels = [c for c in data_df.columns if "label_" in c]
            num_labels = len(labels)
        elif params.task_type == "regression":
            num_labels = 1
        else:
            num_labels = data_df["label"].nunique()

            # Workaround for regression problems
            if params.task_type == "regression":
                data_df["label"] = data_df["label"].astype(float)

        splits_filename = f"{params.dataset_name}.*.csv"
        logger.info("Loading the splits file named %s...", splits_filename)
        splits_df = pd.read_csv(glob(f"{SPLITS_BASE_PATH}/{splits_filename}")[0])
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Defining the source of the model
    if params.base_model_path:
        model_id = os.path.join(PRETRAINED_MODELS_BASE_PATH, params.base_model_path)
    elif params.base_model_name and (params.base_model_name in hf_models):
        base_model_config = hf_models[params.base_model_name]
        model_id = f"{base_model_config['org_id']}/{base_model_config['model_id']}"
    else:
        logger.error("Model %s does not exist in model directory", params.base_model_name)
        sys.exit(1)

    # Defining the source of the tokenizer
    if params.tokenizer_path:
        tokenizer_id = os.path.join(PRETRAINED_TOKENIZERS_BASE_PATH, params.tokenizer_path)
    elif params.tokenizer_name and (params.tokenizer_name in hf_models):
        tokenizer_config = hf_models[params.tokenizer_name]
        tokenizer_id = f"{tokenizer_config['org_id']}/{tokenizer_config['model_id']}"
    else:
        tokenizer_id = model_id

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if params.task_type == "multilabel":
        def tokenize(x):
            example = tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=params.max_length
            )
            example["labels"] = [float(v) for k, v in x.items() if k.startswith("label_")]
            return example
    else:
        def tokenize(x):
            return tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=params.max_length
            )

    logger.info("- - - - -")
    logger.info("Splitting the data...")
    train_df = data_df.loc[splits_df["fold_1"] != 2]
    test_df = data_df.loc[splits_df["fold_1"] == 2]

    logger.info("Transforming the data to a HF dataset...")
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train_df)
    dataset["test"] = Dataset.from_pandas(test_df)

    logger.info("Tokenizing train and test datasets...")
    tokenized_datasets = dataset.map(tokenize)

    # Workaround for regression problems
    if params.task_type == "regression":
        tokenized_datasets["train"] = tokenized_datasets["train"].cast(
            Features({**tokenized_datasets["train"].features, "label": Value("float32")})
        )
        tokenized_datasets["test"] = tokenized_datasets["test"].cast(
            Features({**tokenized_datasets["test"].features, "label": Value("float32")})
        )

    # Model to fine-tune
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type=problem_type,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if params.flash_attention else None,
        use_safetensors=not params.not_use_safetensors
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=results_base_path,
        eval_strategy=IntervalStrategy.STEPS,
        per_device_train_batch_size=params.batch_size_per_gpu,
        per_device_eval_batch_size=params.batch_size_per_gpu,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        eval_accumulation_steps=params.eval_accumulation_steps,  # Useful when eval dataset is large
        learning_rate=params.learning_rate,
        weight_decay=0.01,
        num_train_epochs=params.epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_strategy=IntervalStrategy.STEPS,
        logging_first_step=True,
        logging_steps=0.1,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=0.1,
        save_total_limit=1,
        seed=42,
        bf16=True,
        eval_steps=0.1,
        run_name=params.base_model_name or params.base_model_path,
        load_best_model_at_end=True,
        deepspeed="./deepspeed_config.json",
        # optim="adamw_apex_fused",  # Unable when using deepspeed
        # torch_compile=True,  # Useful, but incompatible with DeepSpeed
        gradient_checkpointing=params.gradient_checkpointing
    )

    # Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # Simple strategy to speed up evaluation on large datasets
        # eval_dataset=tokenized_datasets["test"].train_test_split(test_size=0.01, seed=42)["test"],
        compute_metrics=compute_metrics,  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Fine-tuning the %s model...", model_id)
    trainer.train(params.resume_from_checkpoint)

    logger.info("Reporting error metrics for the train and test datasets...")
    trainer.save_metrics(
        split="train",
        metrics=trainer.evaluate(tokenized_datasets["train"]),  # type: ignore
        combined=False
    )
    trainer.save_metrics(
        split="test",
        metrics=trainer.evaluate(tokenized_datasets["test"]),  # type: ignore
        combined=False
    )

    logger.info("Running inferences on the test dataset...")
    pipe = pipeline(
        "text-classification",
        model=trainer.model,  # type: ignore
        tokenizer=tokenizer,
        batch_size=params.batch_size_per_gpu
    )  # type: ignore
    outputs = pipe(list(dataset["test"]["text"]), top_k=None, truncation=True)
    outputs_df = pd.DataFrame([
        {trainer.model.config.label2id[target["label"]]: target["score"] for target in output}
        for output in outputs
    ])
    outputs_df.to_csv(os.path.join(results_base_path, "test_predictions.csv"), index=False)


if __name__ == "__main__":
    start_time = time.time()
    args = handle_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
