#!/bin/bash
set -e
cd "$(dirname "$0")"

IMAGES="base_512x512.png base1024x1024.png base_2048x2048.png"
KSIZES="3 5"
MODES="0 1"
ITER=100

rm -f resultados_cuda.csv

echo "=== Benchmarks CUDA ==="
for img in $IMAGES; do
    for k in $KSIZES; do
        for m in $MODES; do
            mode_name=$( [ "$m" = "1" ] && echo "shared" || echo "global" )
            echo ""
            echo ">>> $img | k=$k | $mode_name"
            ./cuda_blur "$img" $ITER $k $m
        done
    done
done

echo ""
echo "=== CSV gerado: resultados_cuda.csv ==="
cat resultados_cuda.csv
