#!/bin/bash
set -e

echo "Starting EC2 Data Preparation..."

echo "1. Loading TSV to Parquet..."
python3 code/business_entity_resolution/src/s0_eda.py --load-only

echo "2. Running Normalization..."
python3 code/business_entity_resolution/src/s1_normalize.py

echo "EC2 Data Preparation Complete!"
