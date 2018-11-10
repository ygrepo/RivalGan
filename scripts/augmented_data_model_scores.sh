#!/bin/bash
echo "Augmented data model scores"


python -W ignore insight/ai/pipeline.py --aug_model_scores true --AUGMENTED_DATA_SIZE 5000 --TOTAL_TRAINING_STEPS 6300 --GEN_FILENAME REF