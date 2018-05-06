#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

mkdir -p $EXP_DIR

# Download and preprocess SQuAD data and save in data/
mkdir -p "$DATA_DIR"
rm -rf "$DATA_DIR"
python "$CODE_DIR/preprocessing/squad_preprocess.py" --data_dir "$DATA_DIR"

# Download GloVe vectors to data/
python "$CODE_DIR/preprocessing/download_wordvecs.py" --download_dir "$DATA_DIR"
