# Evaluation

This folder includes all the scripts to fine-tune/prompt and evaluate the models on the different SELU tasks.

## Split the data

The script `split_data.sh` is intended to split a dataset while controlling experimentation elements such as the percentage of instances reserved for testing and the number of folds for cross-validation. The resulting file is a CSV with the same number rows of the source dataset but with columns corresponding to a fold. The values represent whether the instance is assigned to training (1), validation (0) or evaluation (2).

NOTE: Although the implementation partially supports cross-validation, results are only reported on the evaluation subset because of our compute budget.

Be aware of these paths where the pre-processed datasets are read and the split files are stored:

```python
DATASETS_BASE_PATH = "../preprocessing/datasets"
SPLITS_BASE_PATH = "./splits"
```

## Models to evaluate

This is a list of the open-source and proprietary LLMs selected for evaluation. The scripts `run_ft_{task_id}.sh` and `prompt_lm.py` run the fine-tuning and prompting exeriments according to the parameters configured for each task.

|     **Model**     | **Size** | **Architecture** | **Domain adaptation** | **License** |
|:-----------------:|:--------:|:----------------:|:---------------------:|:-----------:|
|     BERT base     |   110m   |   encoder-only   |       Generalist      | Open-source |
|     BERT large    |   340m   |   encoder-only   |                       |             |
|    RoBERTa base   |   125m   |   encoder-only   |                       |             |
|   RoBERTa large   |   355m   |   encoder-only   |                       |             |
|  ModernBERT base  |   150m   |   encoder-only   |                       |             |
|  ModernBERT large |   396m   |   encoder-only   |                       |             |
|    GPT-2 small    |   117m   |   decoder-only   |                       |             |
|    GPT-2 medium   |   345m   |   decoder-only   |                       |             |
|    GPT-2 large    |   774m   |   decoder-only   |                       |             |
|      GPT-2 xl     |   1.5b   |   decoder-only   |                       |             |
|    Llama 3.2 1b   |    1b    |   decoder-only   |                       |             |
|    Llama 3.2 3b   |    3b    |   decoder-only   |                       |             |
|      T5 small     |   60.5m  |  encoder-decoder |                       |             |
|      T5 base      |   223m   |  encoder-decoder |                       |             |
|      T5 large     |   738m   |  encoder-decoder |                       |             |
|       T5 3b       |    3b    |  encoder-decoder |                       |             |
|   CodeBERT base   |   125m   |   encoder-only   |     Domain-adapted    | Open-source |
|    CodeLlama 7b   |    7b    |   decoder-only   |                       |             |
|   StarCoder2 3b   |    3b    |   decoder-only   |                       |             |
|   StarCoder2 7b   |    7b    |   decoder-only   |                       |             |
|    CodeT5+ 220m   |   220m   |  encoder-decoder |                       |             |
|   CodeT5+ 7700m   |   770m   |  encoder-decoder |                       |             |
|       GPT-4o      |     -    |   decoder-only   |       Generalist      | Proprietary |
| Claude 3.5 Sonnet |     -    |   decoder-only   |                       |             |

Finally, the script `train_baseline.py` runs the training and model selection processes for building the baselines according to the parameters configured.

## Results

The subfolder results includes performace scores calculated for all models on each task as follows:

- finetuning: Scores for fine-tuning open-source LLMs.
- prompting: Scores for prompting proprietary LLMs.
- baselines: Scores for the FastText and TF-IDF+XGBoost baselines.
