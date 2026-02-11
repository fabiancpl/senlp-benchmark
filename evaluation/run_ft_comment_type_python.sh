#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# v3 - lr 1e-4
# v4 - lr 1e-5

DATASET=comment_type_python
VERSION=v4
TASK_ARGS="--task_type multilabel --version $VERSION"
TRAINING_ARGS="--epochs 10 --batch_size_per_gpu 16 --gradient_accumulation_steps 1 --learning_rate 1e-5"
TRAINING_ARGS_8x2="--epochs 10 --batch_size_per_gpu 8 --gradient_accumulation_steps 2 --learning_rate 1e-5"
TRAINING_ARGS_4x4="--epochs 10 --batch_size_per_gpu 4 --gradient_accumulation_steps 4 --learning_rate 1e-5"
TRAINING_ARGS_2x8="--epochs 10 --batch_size_per_gpu 2 --gradient_accumulation_steps 8 --learning_rate 1e-5"
TRAINING_ARGS_1x16="--epochs 10 --batch_size_per_gpu 1 --gradient_accumulation_steps 16 --learning_rate 1e-5"


export WANDB_PROJECT="senlp_finetuning"
export WANDB_TAGS=$DATASET,$VERSION


# BERT

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-base-se-cpt/512/3.3B \
#     --tokenizer_name bert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-base-se-cpt/512/11.7B \
#     --tokenizer_name bert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-base-se-fs/512/3.3B \
#     --tokenizer_path bert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-base-se-fs/512/11.7B \
#     --tokenizer_path bert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name bert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-large-se-cpt/512/3.3B \
#     --tokenizer_name bert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path bert-large-se-fs/512/3.3B \
#     --tokenizer_path bert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# RoBERTa

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-base-se-cpt/512/3.3B \
#     --tokenizer_name roberta-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-base-se-cpt/512/11.7B \
#     --tokenizer_name roberta-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-base-se-fs/512/3.3B \
#     --tokenizer_path roberta-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-base-se-fs/512/11.7B \
#     --tokenizer_path roberta-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name roberta-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-large-se-cpt/512/3.3B \
#     --tokenizer_name roberta-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path roberta-large-se-fs/512/3.3B \
#     --tokenizer_path roberta-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# ModernBERT

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-base-se-cpt/8192/3.3B \
#     --tokenizer_name modernbert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-base-se-cpt/8192/9.0B \
#     --tokenizer_name modernbert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-base-se-fs/1024/3.3B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-base-se-fs/1024/7.7B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-base-se-fs/8192/1.4B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name modernbert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-large-se-cpt/8192/2.9B \
#     --tokenizer_name modernbert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-large-se-cpt/8192/3.3B \
#     --tokenizer_name modernbert-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-large-se-fs/1024/2.5B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-large-se-fs/8192/0.4362B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path modernbert-large-se-fs/8192/0.8283B \
#     --tokenizer_path modernbert-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# CodeBERT

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name codebert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path codebert-base-se-cpt/512/3.3B \
#     --tokenizer_name codebert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path codebert-base-se-cpt/512/11.7B \
#     --tokenizer_name codebert-base \
#     $TRAINING_ARGS \
#     $TASK_ARGS



# GPT-2

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name gpt2-small \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-small-se-cpt/1024/3.3B \
#     --tokenizer_name gpt2-small \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-small-se-cpt/1024/11.8B \
#     --tokenizer_name gpt2-small \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-small-se-fs/1024/3.3B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-small-se-fs/1024/11.8B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name gpt2-medium \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-medium-se-cpt/1024/3.3B \
#     --tokenizer_name gpt2-medium \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-medium-se-fs/1024/3.3B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name gpt2-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-large-se-cpt/1024/1.4B \
#     --tokenizer_name gpt2-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-large-se-cpt/1024/3.3B \
#     --tokenizer_name gpt2-large \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-large-se-fs/1024/1.4B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-large-se-fs/1024/3.3B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name gpt2-xl \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-xl-se-cpt/1024/0.6785B \
#     --tokenizer_name gpt2-xl \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-xl-se-cpt/1024/3.3B \
#     --tokenizer_name gpt2-xl \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-xl-se-fs/1024/0.6785B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path gpt2-xl-se-fs/1024/3.3B \
#     --tokenizer_path gpt2-se-fs \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS



# Llama 3.2

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name llama3.2-1b \
#     --max_length 8192 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-1b-se-cpt/8192/1.0B \
#     --tokenizer_name llama3.2-1b \
#     --max_length 8192 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-1b-se-cpt/8192/3.3B \
#     --tokenizer_name llama3.2-1b \
#     --max_length 8192 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-1b-se-fs/1024/0.8745B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 1024 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-1b-se-fs/8192/0.1543B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 8192 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-1b-se-fs/8192/2.4B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 8192 \
#     $TRAINING_ARGS_4x4 \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name llama3.2-3b \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-3b-se-cpt/8192/0.3552B \
#     --tokenizer_name llama3.2-3b \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-3b-se-cpt/8192/3.3B \
#     --tokenizer_name llama3.2-3b \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-3b-se-fs/1024/0.3019B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 1024 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-3b-se-fs/8192/0.0533B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_path llama3.2-3b-se-fs/8192/3.0B \
#     --tokenizer_path llama3.2-se-fs \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing



# CodeLlama

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name codellama-7b \
#     --max_length 8192 \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS \
#     --gradient_checkpointing



# StarCoder2

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name starcoder2-3b \
#     --max_length 8192 \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --gradient_checkpointing


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name starcoder2-7b \
#     --max_length 8192 \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS \
#     --gradient_checkpointing



# T5

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-small \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-base \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-large \
#     --max_length 512 \
#     $TRAINING_ARGS \
#     $TASK_ARGS


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name t5-3b \
#     --max_length 512 \
#     $TRAINING_ARGS_8x2 \
#     $TASK_ARGS



# CodeT5+

# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name codet5p-220m \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --not_use_safetensors


# deepspeed finetune_lm.py \
#     --dataset_name $DATASET \
#     --base_model_name codet5p-770m \
#     $TRAINING_ARGS \
#     $TASK_ARGS \
#     --not_use_safetensors
