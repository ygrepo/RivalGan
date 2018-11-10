#!/bin/bash
echo "Plot data distributions"


python -W ignore insight/ai/pipeline.py --generate_distribution_plot true --AUGMENTED_DATA_SIZE 5000 --TOTAL_TRAINING_STEPS 6300 --GEN_FILENAME REF
