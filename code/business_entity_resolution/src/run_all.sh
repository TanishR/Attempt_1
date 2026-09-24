#!/bin/bash
set -e
echo "Starting pipeline..."
python3 code/business_entity_resolution/src/s0_eda.py
python3 code/business_entity_resolution/src/s1_normalize.py
python3 code/business_entity_resolution/src/s2_embed.py
python3 code/business_entity_resolution/src/s3_block.py
python3 code/business_entity_resolution/src/s4_features.py
python3 code/business_entity_resolution/src/s5_train.py
python3 code/business_entity_resolution/src/s6_tune.py
python3 code/business_entity_resolution/src/s7_predict.py
python3 code/business_entity_resolution/src/s8_write.py
echo "Pipeline finished."
