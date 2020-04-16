#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# philly
export PT_DATA_DIR=/mnt/alfred-data/
export PT_OUTPUT_DIR=/mnt/alfred-data/
export ALFRED_ROOT=$CONFIG_DIR

MODEL=$1       # seq2seq_nl_baseline
EPOCH=$2       # 50
SUFFIX=$3      # high_level_instr
DATA=$4        # json_feat_2.1.0
SPLITS=$5      # apr13.json
EXTRAS=${@:6}  #other flags

export DOUT=$PT_OUTPUT_DIR/exp/model:$MODEL,name:v1_epoch_${EPOCH}_${SUFFIX}
mkdir $DOUT
echo 'Script is making directory '${DOUT}

command="python -u models/train/train_seq2seq.py --model $MODEL --dout $DOUT --data $PT_DATA_DIR/$DATA --splits $PT_DATA_DIR/splits/$SPLITS --gpu --dec_teacher_forcing --epoch $EPOCH $EXTRAS"

echo $command
$command