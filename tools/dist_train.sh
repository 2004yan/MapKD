#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

export LD_LIBRARY_PATH=/opt/conda/envs/map/lib:$LD_LIBRARY_PATH


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
