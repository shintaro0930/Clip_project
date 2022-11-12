#!/bin/bash

if [ $image = "*.jpg" ]; then
    mv $image ${image%.jpg}.png
elif [ $image = "*.png" ]; then
    echo "success"
else
    echo "論外"
fi


#sedは置換後の出力のみで、書き換えは行わない
#sed s/置換対象文字列/置換後文字列/ 指定するファイル名
