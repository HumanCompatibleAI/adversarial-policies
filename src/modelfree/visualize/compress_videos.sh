#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <in dir> <out dir>"
    exit -1
fi

IN_DIR=$1
OUT_DIR=$2

fnames=""
for in_path in ${IN_DIR}/*.mp4; do
    fnames="${fnames} $(basename -s .mp4 ${in_path})"
done

FFMPEG_COMMAND="ffmpeg -i ${IN_DIR}/{prefix}.mp4 -c:v libx264 -preset slow -crf 28"

# These were tuned for my machine. See benchmark_ffmpeg.sh to choose reasonable values.
# Generally there are diminishing returns to using more threads per video.
# Since we have a large number of videos, favor large job count and small thread count.
parallel --header : -j 50% ${FFMPEG_COMMAND} -threads 2 \
                           ${OUT_DIR}/{prefix}_1080p.mp4 ::: prefix ${fnames}
parallel --header : -j 100% ${FFMPEG_COMMAND} -threads 1 -s 1280x720 \
                            ${OUT_DIR}/{prefix}_720p.mp4 ::: prefix ${fnames}
parallel --header : -j 100% ${FFMPEG_COMMAND} -threads 1 -s 854x480 \
                            ${OUT_DIR}/{prefix}_480p.mp4 ::: prefix ${fnames}
