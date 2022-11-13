#!/bin/sh

dir_path="/work/pictures/*"
dirs=`find $dir_path -type f \( -name "*.png" -o -name "*.HEIC" \)`

for image in $dirs;
do
    echo $image
    # ここから実行処理を記述
    if [ $image = "*.jpg" ]; then
    mv $image ${image%.jpg}.png
    elif [ $image = "*.png" ]; then
        echo "success"
    else
        echo "論外"
    fi
done