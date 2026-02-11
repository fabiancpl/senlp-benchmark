#!/bin/bash
set -e


# Binary classification
python split_data.py --dataset_name bug_issue
python split_data.py --dataset_name functional_requirement --k 5
python split_data.py --dataset_name incivility --k 5
python split_data.py --dataset_name quality_requirement --k 5
python split_data.py --dataset_name safety_issue --k 5
python split_data.py --dataset_name security_requirement --k 5
python split_data.py --dataset_name tone_bearing --k 5


# Multi-class classification
python split_data.py --dataset_name closed_question
python split_data.py --dataset_name commit_intent --k 5
python split_data.py --dataset_name issue_intention --k 5
python split_data.py --dataset_name issue_type
python split_data.py --dataset_name question_quality
python split_data.py --dataset_name review_type --k 5
python split_data.py --dataset_name sentiment


# Multi-label classification
python split_data.py --dataset_name comment_type_java --k 5 --cv_strategy iterative
python split_data.py --dataset_name comment_type_pharo --k 5 --cv_strategy iterative
python split_data.py --dataset_name comment_type_python --k 5 --cv_strategy iterative
python split_data.py --dataset_name review_aspect --k 5 --cv_strategy iterative
python split_data.py --dataset_name smell_doc --k 5 --cv_strategy iterative


# Regression
python split_data.py --dataset_name story_points --no_stratify
