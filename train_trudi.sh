#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/opencode/trudi_full.log
set -o pipefail
setsid nohup bash -c 'PYTHONPATH="/data/8hirsch/mmocr:/data/8hirsch/mmocr/projects/ABCNet" /data/8hirsch/mmocr/.venv/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=29507 /data/8hirsch/mmocr/tools/train.py /data/8hirsch/mmocr/projects/ABCNet/config/abcnet/abcnet_resnet50_fpn_500e_trudi_24gbRAM.py --launcher pytorch --work-dir /data/8hirsch/mmocr/work_dirs/abcnet_trudi_full > /tmp/opencode/trudi_full.log 2>&1' < /dev/null &
disown
echo "training launched in background; log: /tmp/opencode/trudi_full.log"