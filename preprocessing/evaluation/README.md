# Pre-processing of evaluation datasets

This ETL orchestrates all the required steps to prepare the data previous to tokenization. The ETL reads the selected dataset in parquet format, applies the pre-processing pipeline and stores it again in the new path.

Additionally, a subprocess to generate diffs of random samples for manual verification is also included. These diff files are stored in HTML format for easy.

Be aware of these paths where the datasets are read and stored:

```python
DATASETS_BASE_PATH = "../../datasets/evaluation"
DIFFS_BASE_PATH = "./diffs"
NEW_DATASETS_BASE_PATH = "./datasets"
```

Note that the NER and MLM tasks are not considered for this stage. However, the datasets need to be available in `DATASETS_BASE_PATH` to be used in subsequent stages (i.e., fine-tuning).
