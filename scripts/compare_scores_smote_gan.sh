#!/bin/bash
echo "Running SMOTE vs GAN scores report"


python -W ignore insight/ai/pipeline.py --compare_scores true --GEN_FILENAME REF --AUGMENTED_DATA_SIZE 100000 --SAMPLER SMOTETomek
