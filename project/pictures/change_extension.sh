for filenamee in *.*; do
    mv $filename ${filename%.*}.png;
done