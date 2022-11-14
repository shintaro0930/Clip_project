#!/bin/bash
dir_path="/work/pictures_2/*"
dirs=`find $dir_path -type f \( -name "*.png" -o -name "*.HEIC" \)`

for image in $dirs;
do
    echo $image
    # ここから実行処理を記述
    if [ $image = "*.jpg" ]; then
    convert *.jpg
    elif [ $image = "*.png" ]; then
        echo "success"
    else
        echo "論外"
    fi
done