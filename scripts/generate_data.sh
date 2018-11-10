#!/bin/bash
echo "Generating new data"


python -W ignore insight/ai/pipeline.py --generate_data true
