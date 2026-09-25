#!/bin/bash
set -e

SPLIT_FILE="cache/split.parquet"

echo "Embedding Train Source 1..."
python3 code/business_entity_resolution/src/s2_embed.py --split train --source s1 --ids-file $SPLIT_FILE

echo "Embedding Train Source 2..."
python3 code/business_entity_resolution/src/s2_embed.py --split train --source s2

echo "Embedding Train Source 3..."
python3 code/business_entity_resolution/src/s2_embed.py --split train --source s3

echo "Embedding Test Source 1..."
python3 code/business_entity_resolution/src/s2_embed.py --split test --source s1

echo "Embedding Test Source 2..."
python3 code/business_entity_resolution/src/s2_embed.py --split test --source s2

echo "Embedding Test Source 3..."
python3 code/business_entity_resolution/src/s2_embed.py --split test --source s3

echo "All embedding completed!"
