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
GPUS=1
SAVE_DIR="./models"
ROOT_DIR="./dataset"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    -b | --BATCH_SIZE)
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
    --gpus)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        GPUS="$2"
        shift # past argument
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
    --SAVE_DIR)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        SAVE_DIR="$2"
        shift # past argument
        ;;
    --ROOT_DIR)
        if [ -z "$2" ]; then
            echo "Missing argument for $key"
            exit 0
        fi
        ROOT_DIR="$2"
        shift # past argument
        ;;
    *)        # unknown option
        shift # past argument
        ;;
    esac
    shift # past value
done

opt="${opt} --batch-size ${BATCH_SIZE} --lr ${LR} --epochs ${EPOCHS} --seed ${SEED} --gpus ${GPUS} --save-dir ${SAVE_DIR} --root-dir ${ROOT_DIR}"

for i in "${arr[@]}"; do
    echo "Running system $i"
    python Experiment.py $opt --system $i
    echo "Finished system $i"
done
