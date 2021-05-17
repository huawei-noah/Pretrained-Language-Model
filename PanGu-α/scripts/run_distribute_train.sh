#!/bin/bash

execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
export RANK_SIZE=$1
export DATASET=$2
export RANK_TABLE_FILE=$3
export MODE=$4
for((i=0;i<$RANK_SIZE;i++));
do
  rm -rf ${execute_path}/device_$i/
  mkdir ${execute_path}/device_$i/
  cd ${execute_path}/device_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../run_pangu_alpha_train.py --data_url=$DATASET --mode=$MODE >train_deep$i.log 2>&1 &
done
