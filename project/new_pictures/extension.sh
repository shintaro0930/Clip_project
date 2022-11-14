#!/bin/bash

for image in `./*`;
do
    echo $image
    # ここから実行処理を記述
    if [ $image = "*.heic" ]; then
        `sudo heif-convert *.heic *.png`
    elif [ $image = "*.HEIC" ]; then
        `sudo heif-convert *.HEIC *.jpg`
    else
        echo "論外"
    fi
done