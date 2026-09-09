#!/usr/bin/env bash
set -euo pipefail

cd /data/8hirsch/mmocr

PYTHONPATH="/data/8hirsch/mmocr" \
  /data/8hirsch/mmocr/.venv/bin/python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29508 \
  tools/train.py \
    configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_trudi.py \
    --launcher pytorch \
    --work-dir work_dirs/dbnetpp_trudi_full
