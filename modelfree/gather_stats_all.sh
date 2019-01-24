
parallel --delay 1 --load 100% --noswap --results logs/ --header : bash single_gather_stats.sh {file} {num} $3 ::: file $1/* ::: num $2 
