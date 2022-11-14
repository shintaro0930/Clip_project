#!/bin/bash

for image in *;
do
    echo $image
    # ここから実行処理を記述
    if [ "$image" = `*.heic` ]; then
        `heif-convert *.heic *.png`
    elif [ "$image" = `*.HEIC` ]; then
        `heif-convert *.HEIC *.jpg`
    else
        echo "論外"
    fi
done