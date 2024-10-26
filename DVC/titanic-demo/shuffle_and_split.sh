#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH1 OUTPUT_PATH2"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH1=$2
OUTPUT_PATH2=$3

tail -n +2 "$INPUT_PATH" | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:42 -nosalt </dev/zero 2>/dev/null) > data_shuffled.csv

total_lines=$(wc -l < data_shuffled.csv)
lines_20p=$((total_lines * 2 / 10))

head -n 1 "$INPUT_PATH" > header.csv
head -n "$lines_20p" data_shuffled.csv > split_20p.csv

cat header.csv split_20p.csv > "$OUTPUT_PATH1"

tail -n +$((lines_20p + 1)) data_shuffled.csv > split_80p.csv

cat header.csv split_80p.csv > "$OUTPUT_PATH2"

rm header.csv data_shuffled.csv split_20p.csv split_80p.csv
