#!/bin/bash
echo "Running SVM vs GAN scores report"


python -W ignore insight/ai/pipeline.py --compare_scores true  --CLASSIFIER SVC --GEN_FILENAME REF --AUGMENTED_DATA_SIZE 100000