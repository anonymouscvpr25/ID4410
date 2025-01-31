#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo "Port:" $PORT

GPUS=4

CONFIG=projects/configs/unipad/uvtr_cam_vs0.075_pretrain_wo_depth.py
echo "Start pretraining with config:" $CONFIG

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --no-validate

CONFIG=projects/configs/unipad_final/final_uvtr_cam_vs0.075_finetune_wo_depth.py
echo "Start finetune..."

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch
