#!/usr/bin/bash

for f in samples/raw/*.mp4; do
    python main.py $f
done