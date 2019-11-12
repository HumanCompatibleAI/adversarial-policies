#!/usr/bin/env bash

for resolution in 1920x1080 1280x720 854x480; do
    for threads in 1 2 4 6 8 12; do
        echo "*** RESOLUTION ${resolution} with THREADS ${threads}"
        time (ffmpeg -y -i $1 -s ${resolution} -c:v libx264 -preset slow -crf 28 -threads ${threads} /tmp/ffmpeg_benchmark.mp4 >/dev/null 2>&1)
    done
done
