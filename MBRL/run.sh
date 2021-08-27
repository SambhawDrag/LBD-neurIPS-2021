#!/bin/bash

declare -a arr=(
    "great-piquant-bumblebee"
    "great-bipedal-bumblebee"
    "great-impartial-bumblebee"
    "great-proficient-bumblebee"
    "lush-piquant-bumblebee"
    "lush-bipedal-bumblebee"
    "lush-impartial-bumblebee"
    "lush-proficient-bumblebee"
    "great-devious-beetle"
    "great-vivacious-beetle"
    "great-mauve-beetle"
    "great-wine-beetle"
    "rebel-devious-beetle"
    "rebel-vivacious-beetle"
    "rebel-mauve-beetle"
    "rebel-wine-beetle"
    "talented-ruddy-butterfly"
    "talented-steel-butterfly"
    "talented-zippy-butterfly"
    "talented-antique-butterfly"
    "thoughtful-ruddy-butterfly"
    "thoughtful-steel-butterfly"
    "thoughtful-zippy-butterfly"
    "thoughtful-antique-butterfly"
)

BATCH_SIZE=32
LR=0.001
EPOCHS=100
SEED=11
CUDA=0
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
ROOT_DIR="./dataset"
NUM_WORKERS=4
VERBOSE=0

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    -b | --batch_size)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        BATCH_SIZE="$2"
        shift # past argument
        ;;
    -lr | --lr)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        LR="$2"
        shift # past argument
        ;;
    --cuda)
        CUDA=1
        ;;
    --verbose)
        VERBOSE=1
        ;;
    --epochs)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        EPOCHS="$2"
        shift # past argument
        ;;
    --seed)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        SEED="$2"
        shift # past argument
        ;;
    --save_dir)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        SAVE_DIR="$2"
        shift # past argument
        ;;
    --log_dir)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        LOG_DIR="$2"
        shift # past argument
        ;;
    --root_dir)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        ROOT_DIR="$2"
        shift # past argument
        ;;
    --num_workers)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        NUM_WORKERS="$2"
        shift
        ;;
    *)        # unknown option
        shift # past argument
        ;;
    esac
    shift # past value
done

opt="${opt} --batch_size ${BATCH_SIZE} --lr ${LR} --epochs ${EPOCHS} --seed ${SEED} --save_dir ${SAVE_DIR} --root_dir ${ROOT_DIR} --num_workers ${NUM_WORKERS}"

if [ $CUDA -eq 1 ]; then
    opt="${opt} --cuda"
fi

if [ $VERBOSE -eq 1 ]; then
    opt="${opt} --verbose"
fi

for i in "${arr[@]}"; do
    echo "Running system $i"
    python main.py $opt --system $i
    echo "Finished system $i"
done
